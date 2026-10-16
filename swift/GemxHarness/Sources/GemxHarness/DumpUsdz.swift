import Foundation
import SceneKit

func dumpUsdz(_ path: String) throws {
    let scene = try SCNScene(url: URL(fileURLWithPath: path), options: [.checkConsistency: false])
    var count = 0, geo = 0, skinners = 0
    func walk(_ n: SCNNode, _ depth: Int) {
        count += 1
        let g = n.geometry
        if g != nil { geo += 1 }
        if let sk = n.skinner { skinners += 1; print(String(repeating: "  ", count: depth) + "[skinner on \(n.name ?? "?")] bones=\(sk.bones.count) first=\(sk.bones.prefix(5).map { $0.name ?? "?" })") }
        let p = n.simdPosition
        let w = n.simdWorldTransform.columns.3
        let e = n.simdEulerAngles
        let tag = g.map { " geo(\($0.sources.first(where: { $0.semantic == .vertex })?.vectorCount ?? 0) v, \($0.materials.count) mat)" } ?? ""
        if true {
            print(String(repeating: "  ", count: depth) + "\(n.name ?? "<nil>")\(tag) pos=(\(String(format: "%.3f %.3f %.3f", p.x, p.y, p.z))) world=(\(String(format: "%.3f %.3f %.3f", w.x, w.y, w.z))) rotDeg=(\(String(format: "%.0f %.0f %.0f", e.x * 57.3, e.y * 57.3, e.z * 57.3))) scale=\(n.simdScale.x)")
        }
        for c in n.childNodes { walk(c, depth + 1) }
    }
    walk(scene.rootNode, 0)
    let (mn, mx) = scene.rootNode.boundingBox
    print("nodes=\(count) geometries=\(geo) skinners=\(skinners) bbox=(\(mn.x),\(mn.y),\(mn.z))→(\(mx.x),\(mx.y),\(mx.z))")
    for name in ["DEF_spine", "DEF_spine_006", "DEF_thigh_L", "DEF_foot_L", "DEF_hand_L", "DEF_hand_R", "DEF_eye_L", "DEF_back_hair", "Pelvis", "Head", "L_Wrist", "mixamorig:Hips", "mixamorig:Head"] {
        if let n = scene.rootNode.childNode(withName: name, recursively: true) {
            let w = n.simdWorldTransform.columns.3
            print(String(format: "KEY %@ world=(%.3f %.3f %.3f)", name, w.x, w.y, w.z))
        }
    }
}
