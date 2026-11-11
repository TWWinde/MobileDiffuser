import Foundation

enum DiffusionModelKind: String, CaseIterable, Hashable, Identifiable {
    case sd3MediumTwoStep
    /// ReSeed paper student (4-step DMD). One set of weights serves both modes: ReSeed alpha 0
    /// is the plain student; alpha > 0 adds the first-interval residual biases (ReSeedBiases.json).
    case sd3MediumFourStep

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .sd3MediumTwoStep:
            return "SD3 Medium 2-Step"
        case .sd3MediumFourStep:
            return "SD3 Medium 4-Step"
        }
    }

    var selectorLabel: String {
        switch self {
        case .sd3MediumTwoStep:
            return "2 steps"
        case .sd3MediumFourStep:
            return "4 steps"
        }
    }

    var shortName: String {
        switch self {
        case .sd3MediumTwoStep:
            return "SD3-2"
        case .sd3MediumFourStep:
            return "SD3-4"
        }
    }

    var stepCount: Int {
        switch self {
        case .sd3MediumTwoStep:
            return 2
        case .sd3MediumFourStep:
            return 4
        }
    }

    var guidanceScale: Float {
        1.0
    }

    var timestepShift: Float {
        3.0
    }

    var resourceFolderName: String {
        resourceFolderName(for: .default)
    }

    func resourceFolderName(for resolution: SD3Resolution) -> String {
        switch self {
        case .sd3MediumTwoStep:
            return resolution.sd3MediumTwoStepResourceFolderName
        case .sd3MediumFourStep:
            return resolution.sd3MediumFourStepResourceFolderName
        }
    }

    /// Selectable ReSeed bias scales (alpha); 0 disables the adapter. Empty = no ReSeed support.
    var reseedScales: [Float] {
        self == .sd3MediumFourStep ? [0, 0.5, 1.0, 1.5] : []
    }
}

enum SD3Resolution: Int, CaseIterable, Hashable, Identifiable {
    case r512 = 512
    /// 1024 uses query-chunked attention, one MMDiT block per Core ML stage (24 body stages),
    /// and a tiled 3x3 decode with the 512 VAE decoder (see Decoder.decodeTiled).
    case r1024 = 1024

    static let `default`: SD3Resolution = .r512

    var id: Int { rawValue }

    var label: String {
        "\(rawValue)x\(rawValue)"
    }

    var shortLabel: String {
        "\(rawValue)"
    }

    private var folderSuffix: String {
        self == .r512 ? "" : "_\(rawValue)"
    }

    var sd3MediumTwoStepResourceFolderName: String {
        "coremlsd3_2step\(folderSuffix)"
    }

    /// 4-step = ReSeed student, one MMDiT block per stage at every resolution (required for the
    /// between-stage ReSeed biases), shared by ReSeed off and on.
    var sd3MediumFourStepResourceFolderName: String {
        "coremlsd3_reseed\(folderSuffix)"
    }
}
