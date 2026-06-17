import Foundation

enum DiffusionModelKind: String, CaseIterable, Hashable, Identifiable {
    case sd3MediumTwoStep
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
}

enum SD3Resolution: Int, Hashable {
    case r512 = 512

    static let `default`: SD3Resolution = .r512

    var label: String {
        "\(rawValue)x\(rawValue)"
    }

    var sd3MediumTwoStepResourceFolderName: String {
        "coremlsd3_2step"
    }

    var sd3MediumFourStepResourceFolderName: String {
        "coremlsd3_4step"
    }
}
