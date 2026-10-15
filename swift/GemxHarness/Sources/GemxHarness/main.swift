import Foundation
import Vision

func die(_ m: String) -> Never { FileHandle.standardError.write(Data((m + "\n").utf8)); exit(1) }
func readF32(_ path: String) throws -> [Float] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
func report(_ motion: BvhMotion, oraclePath: String?) throws {
    print("frames=\(motion.frames.count) frameTime=\(String(format: "%.6f", motion.frameTime))")
    guard let oraclePath else { return }

    var o = try readF32(oraclePath)
    SkeletonTopology.gemxToSceneFrame(&o)
    let all77 = oraclePath.hasSuffix("joints.bin")
    let w = all77 ? JointFrame.floatCount : 66
    let n = min(motion.frames.count, o.count / w)
    var maxErr: Float = 0, sum: Float = 0
    for f in 0..<n {
        let b = all77 ? motion.frames[f].xyz : motion.frames[f].humanml22()
        for i in 0..<w {
            let e = abs(b[i] - o[f * w + i])
            maxErr = max(maxErr, e); sum += e
        }
    }
    print(String(format: "vs oracle (%d frames, %d joints): max %.4f mm, mean %.4f mm", n, w / 3, maxErr * 1000, sum / Float(n * w) * 1000))
}

let args = CommandLine.arguments
guard args.count >= 2 else { die("usage: GemxHarness parse-bvh|hosted|ondevice ...") }

switch args[1] {
case "export-usdz":
    try exportUsdz(args[2], args[3])

case "dump-usdz":
    try dumpUsdz(args[2])

case "parse-bvh":
    let text = try String(contentsOfFile: args[2], encoding: .utf8)
    let t0 = Date()
    let motion = try BvhParser.parse(text)
    print(String(format: "parse %.3f s", Date().timeIntervalSince(t0)))
    try report(motion, oraclePath: args.count > 3 ? args[3] : nil)

case "hosted":
    let client = HostedGemxClient(baseUrl: args[2])
    print("ping:", await client.ping())
    var opts = GemxOptions.defaults
    if ProcessInfo.processInfo.environment["GEMX_DENOISER"] == "onnx" {
        opts = GemxOptions(staticCam: true, noImgfeat: false, hostedDenoiser: .onnx, tier: .fp16, sampleStep: 3, maxSide: 1080)
    }
    let t0 = Date()
    let res = try await client.generate(videoURL: URL(fileURLWithPath: args[3]), options: opts) { p in
        print("  progress: \(p.state) elapsed=\(p.elapsedSeconds)s eta=\(p.etaSeconds.map(String.init) ?? "-") frac=\(p.fraction) | \(p.text)")
    }
    let bvh = res.bvh
    print(String(format: "generate %.1f s, %d bytes; incam BVH: %@; camera: %@", Date().timeIntervalSince(t0), bvh.utf8.count,
                 res.bvhIncam.map { "\($0.utf8.count) bytes" } ?? "none", res.camera.map { "\($0.width)x\($0.height) fx=\($0.fx) cx=\($0.cx) cy=\($0.cy)" } ?? "none"))
    if args.count > 4 { try bvh.write(toFile: args[4], atomically: true, encoding: .utf8) }
    let motion = try BvhParser.parse(bvh)
    try report(motion, oraclePath: nil)
    let f0 = motion.frames[0].xyz, h = SkeletonTopology.head * 3
    print(String(format: "frame0 pelvis=(%.3f %.3f %.3f) head=(%.3f %.3f %.3f)", f0[0], f0[1], f0[2], f0[h], f0[h + 1], f0[h + 2]))
    if let inc = res.bvhIncam, let cam = res.camera {
        if args.count > 4 { try inc.write(toFile: args[4] + ".incam.bvh", atomically: true, encoding: .utf8) }
        let cm = try BvhParser.parse(inc, somaFrame: .camera)
        let c0 = cm.frames[0].xyz
        print(String(format: "incam frame0 (SceneKit cam axes) pelvis=(%.3f %.3f %.3f) head=(%.3f %.3f %.3f), %d frames", c0[0], c0[1], c0[2], c0[h], c0[h + 1], c0[h + 2], cm.frames.count))
        func px(_ x: Float, _ y: Float, _ z: Float) -> String {
            let Z = max(-z, 1e-6); return String(format: "(%.0f, %.0f)", cam.fx * x / Z + cam.cx, cam.fy * (-y) / Z + cam.cy)
        }
        print("  frame0 pelvis px \(px(c0[0], c0[1], c0[2])) head px \(px(c0[45], c0[46], c0[47])) of \(cam.width)x\(cam.height)")
    }

case "ondevice":

    let video = URL(fileURLWithPath: args[2])
    let client = try OnDeviceGemxClient(modelsDirectory: URL(fileURLWithPath: args[3]))
    let oracleDir = args.count > 4 && args[4] != "-" ? args[4] : nil
    var opts = GemxOptions.defaults
    if let s = ProcessInfo.processInfo.environment["GEMX_STEP"], let v = Int(s) {
        opts = GemxOptions(staticCam: true, noImgfeat: false, hostedDenoiser: .pytorch, tier: .fp16, sampleStep: v, maxSide: 1080)
    }
    if let s = ProcessInfo.processInfo.environment["GEMX_THREADS"], let v = Int(s) { client.vitposeThreads = v }

    if ProcessInfo.processInfo.environment["GEMX_NATIVE_FPS"] != nil { opts.resampleFps = nil }
    if ProcessInfo.processInfo.environment["GEMX_POSE"] == "onnx" { opts.poseBackend = .onnx; client.preferredPoseBackend = .onnx }
    if ProcessInfo.processInfo.environment["GEMX_FLIP"] == "0" { opts.flipTest = false }
    if let m = ProcessInfo.processInfo.environment["GEMX_POSE_MODEL"] { opts.poseModel = m }
    if let t = ProcessInfo.processInfo.environment["GEMX_TIER"], let v = GemxTier(rawValue: t) {
        opts = GemxOptions(staticCam: opts.staticCam, noImgfeat: opts.noImgfeat, hostedDenoiser: opts.hostedDenoiser, tier: v, sampleStep: opts.sampleStep, maxSide: opts.maxSide,
                           resampleFps: opts.resampleFps, poseBackend: opts.poseBackend, poseCompute: opts.poseCompute, flipTest: opts.flipTest, poseModel: opts.poseModel)
    }
    if let c = ProcessInfo.processInfo.environment["GEMX_POSE_COMPUTE"], let v = GemxPoseCompute(rawValue: c) { opts.poseCompute = v }
    if let s = ProcessInfo.processInfo.environment["GEMX_DENOISE_WINDOW"], let v = Int(s) { OnDeviceGemxClient.denoiseWindow = v }
    if let s = ProcessInfo.processInfo.environment["GEMX_DENOISE_OVERLAP"], let v = Int(s) { OnDeviceGemxClient.denoiseOverlap = v }
    switch ProcessInfo.processInfo.environment["GEMX_DETECTOR"] {
    case "rect": GemxDetector.mode = .humanRectangle
    case "pose": GemxDetector.mode = .bodyPose
    default: break
    }
    let reader = try await GemxVideoReader(url: video, maxSide: opts.maxSide)
    print("video: \(reader.info.width)x\(reader.info.height) fps=\(reader.info.fps) est=\(reader.info.estimatedFrames) step=\(opts.sampleStep) vpThreads=\(client.vitposeThreads)")

    func cmp(_ a: [Double], _ b: [Float], _ label: String, scale: Double = 1000, unit: String = "mm") {
        let n = min(a.count, b.count)
        var mx = 0.0, sum = 0.0
        for i in 0..<n { let e = abs(a[i] - Double(b[i])); mx = max(mx, e); sum += e }
        print(String(format: "  %@: max %.4f %@, mean %.4f %@ (%d values)", label, mx * scale, unit, sum / Double(n) * scale, unit, n))
    }
    func cmpF(_ a: [Float], _ b: [Float], _ label: String, scale: Double = 1, unit: String = "") {
        cmp(a.map(Double.init), b, label, scale: scale, unit: unit)
    }
    func oracle(_ name: String) throws -> [Float] { try readF32(oracleDir! + "/" + name + ".bin") }
    func profile(_ a: [Double], _ b: [Float], _ label: String) {
        let L = a.count / 231
        var per = [Double](repeating: 0, count: L), pelvis = [Double](repeating: 0, count: L)
        for f in 0..<L {
            var s = 0.0
            for i in 0..<231 { s += abs(a[f * 231 + i] - Double(b[f * 231 + i])) }
            per[f] = s / 231 * 1000
            pelvis[f] = (0..<3).map { abs(a[f * 231 + $0] - Double(b[f * 231 + $0])) }.max()! * 1000
        }
        let worst = per.indices.max { per[$0] < per[$1] }!
        print("  \(label): per-frame mean mm: " + stride(from: 0, to: L, by: max(1, L / 12)).map { String(format: "%d:%.0f", $0, per[$0]) }.joined(separator: " ") + String(format: " | worst frame %d (%.0f mm, pelvis %.0f mm)", worst, per[worst], pelvis[worst]))
    }

    if let oracleDir, ProcessInfo.processInfo.environment["GEMX_ONLY_D"] == nil {
        print("[A] decode/skeleton/FK/ground from oracle pred_x")
        let px = try oracle("pred_x"); let L = px.count / 585
        let pc = (try? oracle("pred_cam")) ?? [Float](repeating: 0, count: L * 3)
        let bbA = try oracle("bbx_xys")
        let camA = GemxCamera.estimated(width: reader.info.width, height: reader.info.height)
        let t0 = Date()
        let fin = client.finish(predX: px, predCam: pc, bbxXys: bbA, camera: camA, L: L, fps: reader.info.fps)
        print(String(format: "  finish %.3f s for %d frames", Date().timeIntervalSince(t0), L))
        cmp(fin.joints77, try oracle("joints"), "joints77 vs oracle")
        let parsed = try BvhParser.parse(fin.bvh)
        var mx: Float = 0
        for f in 0..<L { for i in 0..<JointFrame.floatCount { mx = max(mx, abs(parsed.frames[f].xyz[i] - fin.motion.frames[f].xyz[i])) } }
        print(String(format: "  Swift BVH text → BvhParser → 77 joints vs direct frames: max %.4f mm", mx * 1000))
        if let jc = try? oracle("joints_incam") {
            print("[A2] in-camera motion (overlay) from oracle pred_x + pred_cam + boxes")
            cmp(fin.jointsIncam, jc, "incam joints77 vs oracle")
            let parsedC = try BvhParser.parse(fin.bvhIncam, somaFrame: .camera)
            var mxc: Float = 0
            for f in 0..<L { for i in 0..<JointFrame.floatCount { mxc = max(mxc, abs(parsedC.frames[f].xyz[i] - fin.camera.frames[f].xyz[i])) } }
            print(String(format: "  Swift incam BVH → BvhParser(.camera) → 77 joints vs direct camera frames: max %.4f mm", mxc * 1000))
            let kpA = try oracle("kp2d")
            let proj = GemxPipeline.project(points: fin.jointsIncam, K: camA.K, count: L * 77)
            var sum = 0.0, n = 0, sum22 = 0.0, n22 = 0
            for f in 0..<L { for j in 0..<77 where kpA[(f * 77 + j) * 3 + 2] > 0.3 {
                let dx = proj[(f * 77 + j) * 2] - Double(kpA[(f * 77 + j) * 3]), dy = proj[(f * 77 + j) * 2 + 1] - Double(kpA[(f * 77 + j) * 3 + 1])
                let e = (dx * dx + dy * dy).squareRoot(); sum += e; n += 1
                if GemxConst.humanml22.contains(j) { sum22 += e; n22 += 1 }
            } }
            print(String(format: "  reprojection vs kp2d: mean %.1f px (all joints), %.1f px (22 body joints) in %dx%d", sum / Double(max(n, 1)), sum22 / Double(max(n22, 1)), camA.width, camA.height))
            if args.count > 5 { try fin.bvhIncam.write(toFile: args[5] + ".incam.bvh", atomically: true, encoding: .utf8) }
        }

        print("[B] denoiser from oracle kp2d + boxes (Swift ORT vs Python ORT, same fp16 model)")
        let kp = try oracle("kp2d"), bb = try oracle("bbx_xys")
        let t1 = Date()
        let rB = try client.run(reader: reader, options: opts, oracleBoxes: bb, oracleKp2d: kp)
        print(String(format: "  run %.1f s %@", Date().timeIntervalSince(t1), rB.timings.description))
        cmpF(rB.predX, px, "pred_x vs oracle")
        cmp(rB.joints77, try oracle("joints"), "joints77 vs oracle")

        print("[C] Swift decode + crop + VitPose from oracle boxes")
        let t2 = Date()
        let rC = try client.run(reader: reader, options: opts, oracleBoxes: bb, oracleKp2d: nil)
        print(String(format: "  run %.1f s %@", Date().timeIntervalSince(t2), rC.timings.description))

        let idx = GemxPipeline.sampleIndices(n: rC.kp2d.count / 231, step: opts.sampleStep)
        var mxp: Float = 0, sump: Float = 0, cnt = 0
        for i in idx { for j in 0..<77 {
            let dx = rC.kp2d[i * 231 + j * 3] - kp[i * 231 + j * 3], dy = rC.kp2d[i * 231 + j * 3 + 1] - kp[i * 231 + j * 3 + 1]
            let e = (dx * dx + dy * dy).squareRoot(); mxp = max(mxp, e); sump += e; cnt += 1 } }
        print(String(format: "  kp2d at %d sampled frames: max %.3f px, mean %.3f px", idx.count, mxp, sump / Float(cnt)))
        cmp(rC.joints77, try oracle("joints"), "joints77 vs oracle")
        profile(rC.joints77, try oracle("joints"), "C")
        var perJ = [Float](repeating: 0, count: 77)
        for i in idx { for j in 0..<77 {
            let dx = rC.kp2d[i * 231 + j * 3] - kp[i * 231 + j * 3], dy = rC.kp2d[i * 231 + j * 3 + 1] - kp[i * 231 + j * 3 + 1]
            perJ[j] = max(perJ[j], (dx * dx + dy * dy).squareRoot()) } }
        print("  kp2d max px per joint: " + perJ.enumerated().map { String(format: "%d:%.1f", $0.offset, $0.element) }.joined(separator: " "))
        if args.count > 5 { try rC.bvh.write(toFile: args[5] + ".C.bvh", atomically: true, encoding: .utf8) }
    }

    print("[D] full on-device pipeline (Apple Vision detector)")
    let t3 = Date()
    let r = try client.run(reader: reader, options: opts) { print("  progress: \($0)") }
    print(String(format: "  run %.1f s %@", Date().timeIntervalSince(t3), r.timings.description))
    print("  frames=\(r.motion.frames.count) frameTime=\(r.motion.frameTime)")
    if let oracleDir {
        let bb = try oracle("bbx_xys")
        var mc: Float = 0, ms: Float = 0
        for i in 0..<(r.bbxXys.count / 3) {
            mc = max(mc, abs(r.bbxXys[i * 3] - bb[i * 3]), abs(r.bbxXys[i * 3 + 1] - bb[i * 3 + 1])); ms = max(ms, abs(r.bbxXys[i * 3 + 2] - bb[i * 3 + 2]) / bb[i * 3 + 2])
        }
        print(String(format: "  boxes vs YOLOX oracle: max centre diff %.1f px, max size diff %.1f%%", mc, ms * 100))
        var sdx: Float = 0, sdy: Float = 0, sr: Float = 0, srr: Float = 0
        let nb = r.bbxXys.count / 3
        for i in 0..<nb {
            sdx += r.bbxXys[i * 3] - bb[i * 3]; sdy += r.bbxXys[i * 3 + 1] - bb[i * 3 + 1]
            let ratio = r.bbxXys[i * 3 + 2] / bb[i * 3 + 2]; sr += ratio; srr += ratio * ratio
        }
        let mr = sr / Float(nb)
        print(String(format: "  box stats (swift − yolox): mean dx %.1f px, mean dy %.1f px, size ratio %.3f ± %.3f", sdx / Float(nb), sdy / Float(nb), mr, (srr / Float(nb) - mr * mr).squareRoot()))

        var st: Float = 0, sb: Float = 0
        for i in 0..<nb {
            let hy = bb[i * 3 + 2] / 1.2
            let topS = r.bbxXys[i * 3 + 1] - r.bbxXys[i * 3 + 2] / 2.4, topY = bb[i * 3 + 1] - hy / 2
            let botS = r.bbxXys[i * 3 + 1] + r.bbxXys[i * 3 + 2] / 2.4, botY = bb[i * 3 + 1] + hy / 2
            st += (topS - topY) / hy; sb += (botS - botY) / hy
        }
        print(String(format: "  per-edge (fraction of YOLOX height): top %+.3f, bottom %+.3f", st / Float(nb), sb / Float(nb)))
        try r.bbxXys.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: oracleDir + "/bbx_xys_swift.bin"))
        cmp(r.joints77, try oracle("joints"), "joints77 vs oracle (detector-limited)")
        profile(r.joints77, try oracle("joints"), "D")
    }
    let f0 = r.motion.frames[0].xyz, h0 = SkeletonTopology.head * 3
    print(String(format: "  frame0 pelvis=(%.3f %.3f %.3f) head=(%.3f %.3f %.3f)", f0[0], f0[1], f0[2], f0[h0], f0[h0 + 1], f0[h0 + 2]))
    if args.count > 5 {
        try r.bvh.write(toFile: args[5], atomically: true, encoding: .utf8); print("  wrote \(args[5])")
        try r.bvhIncam.write(toFile: args[5] + ".incam.bvh", atomically: true, encoding: .utf8)
    }
    do {

        let camD = GemxCamera.estimated(width: reader.info.width, height: reader.info.height)
        let proj = GemxPipeline.project(points: r.jointsIncam, K: camD.K, count: r.jointsIncam.count / 3)
        let L = r.jointsIncam.count / 231
        var sum22 = 0.0, n22 = 0
        for f in 0..<L { for j in GemxConst.humanml22 where r.kp2d[(f * 77 + j) * 3 + 2] > 0.3 {
            let dx = proj[(f * 77 + j) * 2] - Double(r.kp2d[(f * 77 + j) * 3]), dy = proj[(f * 77 + j) * 2 + 1] - Double(r.kp2d[(f * 77 + j) * 3 + 1])
            sum22 += (dx * dx + dy * dy).squareRoot(); n22 += 1
        } }
        print(String(format: "  D overlay: in-camera 22-joint reprojection vs own kp2d: mean %.1f px in %dx%d", sum22 / Double(max(n22, 1)), camD.width, camD.height))
    }

case "boxes":

    let client = try OnDeviceGemxClient(modelsDirectory: URL(fileURLWithPath: args[3]))
    let reader = try await GemxVideoReader(url: URL(fileURLWithPath: args[2]), maxSide: 1080)
    let bb = try readF32(args[4] + "/bbx_xys.bin")
    let step = args.count > 5 ? Int(args[5])! : 3
    let modes: [(String, GemxDetector.Mode, (Float, Float, Float))] = [
        ("rect", .humanRectangle, (0, 0, 0)),
        ("pose raw", .bodyPose, (0, 0, 0)),
        ("pose margins", .bodyPose, GemxDetector.margins),
        ("union raw", .union, (0, 0, 0)),
        ("union m0.08/0.06", .union, (0.08, 0.06, 0.04)),
        ("union m0.12/0.10", .union, (0.12, 0.10, 0.04)),
    ]
    for (label, mode, m) in modes {
        GemxDetector.mode = mode; GemxDetector.margins = m
        let t0 = Date()
        let (n, idx, xys) = try client.detectBoxes(reader: reader, step: step, onProgress: nil)
        let dt = Date().timeIntervalSince(t0)
        var st: Float = 0, sb: Float = 0, sdx: Float = 0, sr: Float = 0, srr: Float = 0
        for (k, i) in idx.enumerated() {
            let hy = bb[i * 3 + 2] / 1.2
            let topS = xys[k].y - xys[k].z / 2.4, topY = bb[i * 3 + 1] - hy / 2
            let botS = xys[k].y + xys[k].z / 2.4, botY = bb[i * 3 + 1] + hy / 2
            st += (topS - topY) / hy; sb += (botS - botY) / hy; sdx += (xys[k].x - bb[i * 3]) / hy
            let r = xys[k].z / bb[i * 3 + 2]; sr += r; srr += r * r
        }
        let c = Float(idx.count), mr = sr / c
        print(String(format: "%@: %d/%d frames %.1fs | top %+.3f bottom %+.3f dx %+.3f (of YOLOX height) | size ratio %.3f ± %.3f", label, idx.count, n, dt, st / c, sb / c, sdx / c, mr, (srr / c - mr * mr).squareRoot()))
    }

case "pose":

    let reader = try await GemxVideoReader(url: URL(fileURLWithPath: args[2]), maxSide: 1080)
    let want = Int(args[3])!
    try reader.forEachFrame { f in
        guard f.index == want else { return true }
        guard let pb = f.makePixelBuffer() else { return false }
        let req = VNDetectHumanBodyPoseRequest()
        try VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:]).perform([req])
        print("frame \(f.index) \(f.width)x\(f.height): \(req.results?.count ?? 0) people")
        for obs in req.results ?? [] {
            let pts = try obs.recognizedPoints(.all)
            for (name, p) in pts.sorted(by: { $0.key.rawValue.rawValue < $1.key.rawValue.rawValue }) {
                print(String(format: "  %@ conf %.2f  x %.0f y %.0f", name.rawValue.rawValue, p.confidence, p.location.x * Double(f.width), (1 - p.location.y) * Double(f.height)))
            }
        }
        let fpb = f.makePixelBuffer()!
        let r = GemxDetector.fromHumanRectangle(fpb, .up, f.width, f.height); print("  rect:", r.map { "\($0)" } ?? "nil")
        GemxDetector.margins = (0, 0, 0)
        let b = GemxDetector.fromBodyPose(fpb, .up, f.width, f.height); print("  pose extents:", b.map { "\($0)" } ?? "nil")
        return false
    }

case "vitpose-crop":

    let client = try OnDeviceGemxClient(modelsDirectory: URL(fileURLWithPath: args[2]))
    let crop = try readF32(args[3])
    let hm = try client.runVitPoseOnCrop(crop)
    try hm.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: args[4]))
    print("heatmap floats:", hm.count)

case "dump-frame":

    let reader = try await GemxVideoReader(url: URL(fileURLWithPath: args[2]), maxSide: 1080)
    let want = Int(args[3])!
    var found = false
    try reader.forEachFrame { f in
        guard f.index == want else { return true }
        var d = Data()
        var w = Int32(f.width), h = Int32(f.height)
        d.append(Data(bytes: &w, count: 4)); d.append(Data(bytes: &h, count: 4))
        for y in 0..<f.height { d.append(Data(bytes: f.data + y * f.bytesPerRow, count: f.width * 4)) }
        try d.write(to: URL(fileURLWithPath: args[4]))
        found = true
        return false
    }
    print("video \(reader.info.width)x\(reader.info.height) found=\(found)")

case "dump-crop":

    let reader = try await GemxVideoReader(url: URL(fileURLWithPath: args[2]), maxSide: 1080)
    let want = Int(args[3])!
    let box = SIMD3<Float>(Float(args[4])!, Float(args[5])!, Float(args[6])!)
    try reader.forEachFrame { f in
        guard f.index == want else { return true }
        var crop = [Float](repeating: 0, count: 3 * 256 * 192)
        crop.withUnsafeMutableBufferPointer { GemxPipeline.cropForVitPose(frame: f, box: box, into: $0.baseAddress!) }
        try crop.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: args[7]))
        return false
    }
    print("ok")

default:
    die("unknown command \(args[1])")
}
