import Foundation

enum ModelResourceLocations {
    static var modelsRootURL: URL {
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        return support.appending(path: "MobileDiffuserModels", directoryHint: .isDirectory)
    }
}

@MainActor
final class ModelResourceManager: ObservableObject {
    enum DownloadState: Equatable {
        case idle
        case downloading
        case complete
        case failed(String)
    }

    @Published private(set) var state: DownloadState = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var statusMessage: String = "Models are stored in Application Support."

    private let repoID = "Wenwu2000/MobileDiffuser-SD3-medium"
    private let session = URLSession.shared

    var isDownloading: Bool {
        if case .downloading = state { return true }
        return false
    }

    static var modelsRootURL: URL {
        ModelResourceLocations.modelsRootURL
    }

    func localURL(for folderName: String) -> URL {
        Self.modelsRootURL.appending(path: folderName, directoryHint: .isDirectory)
    }

    func hasResources(for model: DiffusionModelKind, resolution: SD3Resolution = .default) -> Bool {
        SD3PipelineLoader.hasRequiredResources(at: localURL(for: model.resourceFolderName(for: resolution)))
    }

    func downloadAll() async {
        await downloadFolders(DiffusionModelKind.allCases.map { $0.resourceFolderName })
    }

    func downloadSelected(_ model: DiffusionModelKind, resolution: SD3Resolution = .default) async {
        await downloadFolders([model.resourceFolderName(for: resolution)])
    }

    private func downloadFolders(_ folders: [String]) async {
        guard !isDownloading else { return }
        state = .downloading
        progress = 0
        statusMessage = "Preparing download..."

        do {
            try FileManager.default.createDirectory(
                at: Self.modelsRootURL,
                withIntermediateDirectories: true
            )

            var allFiles: [HFFile] = []
            for folder in folders {
                statusMessage = "Reading \(folder) manifest..."
                let files = try await listFiles(in: folder)
                allFiles.append(contentsOf: files)
            }

            guard !allFiles.isEmpty else {
                throw NSError(
                    domain: "MobileDiffuser.ModelResourceManager",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "No model files were found in the Hugging Face repository."]
                )
            }

            let totalBytes = allFiles.compactMap(\.size).reduce(0, +)
            var downloadedBytes: Int64 = 0

            for (index, file) in allFiles.enumerated() {
                let target = Self.modelsRootURL.appending(path: file.path)
                if FileManager.default.fileExists(atPath: target.path),
                   let expectedSize = file.size,
                   (try? target.resourceValues(forKeys: [.fileSizeKey]).fileSize) == Int(expectedSize)
                {
                    downloadedBytes += expectedSize
                    updateProgress(downloadedBytes: downloadedBytes, totalBytes: totalBytes, index: index, count: allFiles.count)
                    continue
                }

                statusMessage = "Downloading \(file.path)"
                try await downloadFile(file.path, to: target)
                if let size = file.size {
                    downloadedBytes += size
                }
                updateProgress(downloadedBytes: downloadedBytes, totalBytes: totalBytes, index: index + 1, count: allFiles.count)
            }

            state = .complete
            progress = 1
            statusMessage = "Model resources are ready."
        } catch {
            state = .failed(error.localizedDescription)
            statusMessage = "Download failed: \(error.localizedDescription)"
        }
    }

    private func updateProgress(downloadedBytes: Int64, totalBytes: Int64, index: Int, count: Int) {
        if totalBytes > 0 {
            progress = min(1, Double(downloadedBytes) / Double(totalBytes))
        } else {
            progress = min(1, Double(index) / Double(max(count, 1)))
        }
    }

    private func listFiles(in folder: String) async throws -> [HFFile] {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/api/models/\(repoID)/tree/main/\(folder)"
        components.queryItems = [URLQueryItem(name: "recursive", value: "1")]

        guard let url = components.url else {
            throw URLError(.badURL)
        }

        let (data, response) = try await session.data(from: url)
        try validateHTTP(response)
        let entries = try JSONDecoder().decode([HFTreeEntry].self, from: data)
        return entries
            .filter { $0.type == "file" }
            .map { HFFile(path: $0.path, size: $0.size) }
    }

    private func downloadFile(_ path: String, to target: URL) async throws {
        try FileManager.default.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(repoID)/resolve/main/\(path)"

        guard let url = components.url else {
            throw URLError(.badURL)
        }

        let (temporaryURL, response) = try await session.download(from: url)
        try validateHTTP(response)

        if FileManager.default.fileExists(atPath: target.path) {
            try FileManager.default.removeItem(at: target)
        }
        try FileManager.default.moveItem(at: temporaryURL, to: target)
    }

    private func validateHTTP(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw NSError(
                domain: "MobileDiffuser.ModelResourceManager",
                code: http.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(http.statusCode)"]
            )
        }
    }
}

private struct HFTreeEntry: Decodable {
    let path: String
    let type: String
    let size: Int64?
}

private struct HFFile {
    let path: String
    let size: Int64?
}
