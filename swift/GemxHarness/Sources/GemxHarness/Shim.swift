import Foundation

enum OnDeviceError: Error, CustomStringConvertible {
    case resourceMissing(String)
    case ortError(String)
    case shapeMismatch(String)
    var description: String {
        switch self {
        case .resourceMissing(let n): return "On-device: bundled resource missing — \(n)"
        case .ortError(let m):        return "On-device: ONNX Runtime error — \(m)"
        case .shapeMismatch(let m):   return "On-device: shape mismatch — \(m)"
        }
    }
}
