import Foundation
import SceneKit

func exportUsdz(_ input: String, _ output: String) throws {
    let scene = try SCNScene(url: URL(fileURLWithPath: input), options: [.checkConsistency: false])
    func skinners(_ s: SCNScene) -> [(String, Int)] {
        var out: [(String, Int)] = []
        func walk(_ n: SCNNode) { if let sk = n.skinner { out.append((n.name ?? "?", sk.bones.count)) }; n.childNodes.forEach(walk) }
        walk(s.rootNode); return out
    }
    print("loaded: skinners=\(skinners(scene))")
    let ok = scene.write(to: URL(fileURLWithPath: output), options: nil, delegate: nil, progressHandler: nil)
    print("write ok=\(ok)")
    let back = try SCNScene(url: URL(fileURLWithPath: output), options: [.checkConsistency: false])
    print("reloaded: skinners=\(skinners(back))")
}
