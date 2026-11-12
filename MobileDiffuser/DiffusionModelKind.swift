import Foundation

/// First-level choice in the UI.
enum ModelFamily: String, CaseIterable, Hashable, Identifiable {
    case sd3Medium
    case flux2Klein

    var id: String { rawValue }

    var label: String {
        switch self {
        case .sd3Medium: return "SD3M"
        case .flux2Klein: return "FLUX.2 Klein"
        }
    }
}

/// A model family at a given step count. The UI picks family and steps independently.
enum DiffusionModelKind: String, CaseIterable, Hashable, Identifiable {
    case sd3MediumTwoStep
    /// ReSeed paper student (4-step DMD). One set of weights serves both modes: ReSeed alpha 0
    /// is the plain student; alpha > 0 adds the first-interval residual biases (ReSeedBiases.json).
    case sd3MediumFourStep
    /// FLUX.2 Klein 4B (4-step distilled). Text encoder (Qwen3), denoiser and VAE are all split
    /// Core ML stages on the ANE (see Flux2KleinPipeline). The 2-step entry runs the same weights
    /// on a 2-step schedule: faster, lower quality.
    case flux2KleinTwoStep
    case flux2KleinFourStep

    var id: String { rawValue }

    static let stepChoices = [2, 4]

    static func kind(family: ModelFamily, steps: Int) -> DiffusionModelKind {
        switch (family, steps) {
        case (.sd3Medium, 2): return .sd3MediumTwoStep
        case (.sd3Medium, _): return .sd3MediumFourStep
        case (.flux2Klein, 2): return .flux2KleinTwoStep
        case (.flux2Klein, _): return .flux2KleinFourStep
        }
    }

    var family: ModelFamily {
        switch self {
        case .sd3MediumTwoStep, .sd3MediumFourStep: return .sd3Medium
        case .flux2KleinTwoStep, .flux2KleinFourStep: return .flux2Klein
        }
    }

    var isKlein: Bool { family == .flux2Klein }

    /// Kinds served by the SD3 pipeline and its shared download button.
    static var sd3Kinds: [DiffusionModelKind] { [.sd3MediumTwoStep, .sd3MediumFourStep] }

    /// One entry per distinct resource folder set (both Klein step counts share their weights).
    static var resourceKinds: [DiffusionModelKind] { [.sd3MediumTwoStep, .sd3MediumFourStep, .flux2KleinFourStep] }

    var supportedResolutions: [SD3Resolution] { SD3Resolution.allCases }

    var stepCount: Int {
        switch self {
        case .sd3MediumTwoStep, .flux2KleinTwoStep: return 2
        case .sd3MediumFourStep, .flux2KleinFourStep: return 4
        }
    }

    var displayName: String {
        switch family {
        case .sd3Medium: return "SD3 Medium \(stepCount)-Step"
        case .flux2Klein: return "FLUX.2 Klein 4B \(stepCount)-Step"
        }
    }

    var selectorLabel: String { "\(stepCount) steps" }

    /// Label of the resource set in Settings.
    var resourceLabel: String {
        isKlein ? "FLUX.2 Klein" : "SD3M \(stepCount)-step"
    }

    var shortName: String {
        isKlein ? "Klein-\(stepCount)" : "SD3-\(stepCount)"
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
        case .flux2KleinTwoStep, .flux2KleinFourStep:
            return "coreml_flux2_klein_\(resolution.rawValue)"
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

/// Thread-safe cancel request shared between the UI and the generation task.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func set(_ newValue: Bool) {
        lock.lock(); value = newValue; lock.unlock()
    }
}
