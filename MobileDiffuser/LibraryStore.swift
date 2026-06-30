// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Durable on-device Library storage.
///
/// The store writes original PNGs plus a small manifest under Application Support. It is an actor so
/// PNG encode/decode and JSON IO never run on AppModel's main actor. Bad records are skipped on load:
/// a corrupt image or stale manifest entry must not prevent the app from launching.
actor LibraryStore {
    private struct Manifest: Codable {
        var version: Int = 1
        var generations: [Record]
    }

    private struct Record: Codable {
        var id: UUID
        var imageFilename: String
        var prompt: String
        var modelID: String
        var modelName: String
        var size: Int
        var steps: Int
        var seed: UInt64
        var duration: TimeInterval
        var settings: [GenerationSetting]
        var date: Date
    }

    private let root: URL
    private let imagesDir: URL
    private let manifestURL: URL
    private let fm = FileManager.default

    init(baseURL: URL) {
        root = baseURL.appending(component: "Library", directoryHint: .isDirectory)
        imagesDir = root.appending(component: "images", directoryHint: .isDirectory)
        manifestURL = root.appending(component: "manifest.json")
    }

    func load() -> [Generation] {
        try? prepareDirectories()
        try? cleanupTemporaryFiles()
        // No manifest → a genuinely empty library (first launch). Safe to return empty.
        guard fm.fileExists(atPath: manifestURL.path) else { return [] }
        do {
            let manifest = try decodeManifest()
            return manifest.generations.compactMap(loadGeneration(from:))
                .sorted { $0.date > $1.date }
        } catch {
            // Manifest is PRESENT but unreadable. Do NOT report an empty library — a subsequent save
            // would then have nothing to keep and (previously) orphan-deleted every PNG, wiping the
            // user's pictures from one transient read error. Back the bad manifest up and recover
            // whatever images are still on disk (metadata is lost, but the images survive).
            try? backupCorruptManifest()
            return recoverFromImages().sorted { $0.date > $1.date }
        }
    }

    /// Move an undecodable manifest aside so the next save() rebuilds a clean one without overwriting
    /// the evidence (and without us mistaking it for "empty").
    private func backupCorruptManifest() throws {
        let backup = root.appending(component: "manifest.corrupt.json")
        if fm.fileExists(atPath: backup.path) { try? fm.removeItem(at: backup) }
        try fm.moveItem(at: manifestURL, to: backup)
    }

    /// Last-resort recovery: rebuild records straight from the PNGs on disk when the manifest is lost.
    private func recoverFromImages() -> [Generation] {
        guard let urls = try? fm.contentsOfDirectory(
            at: imagesDir, includingPropertiesForKeys: [.contentModificationDateKey]) else { return [] }
        return urls.filter { $0.pathExtension.lowercased() == "png" }.compactMap { url in
            guard let image = ImageCodec.cgImage(from: url) else { return nil }
            let filename = url.lastPathComponent
            let id = UUID(uuidString: (filename as NSString).deletingPathExtension) ?? UUID()
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return Generation(id: id, image: image, prompt: "", modelID: "", modelName: "",
                              size: image.width, steps: 0, seed: 0, duration: 0, settings: [],
                              date: date, imageFilename: filename)
        }
    }

    func save(_ generations: [Generation]) throws {
        try prepareDirectories()
        let records = try generations.map { try persist($0) }
        try writeManifest(Manifest(generations: records))
        // NB: save() is intentionally non-destructive — it never prunes images. Deletion happens only
        // in delete() (which removes that one file). Pruning here meant a save() called with a partial
        // history (e.g. after a failed manifest read) would delete every other PNG. At worst a few
        // stale PNGs linger; they are cleaned up by an explicit compaction, never by a routine save.
    }

    /// Explicit, opt-in compaction: remove PNGs not referenced by the given live set. Call this only
    /// from a deliberate "clean up storage" action, never from the save hot path.
    func compact(keeping generations: [Generation]) throws {
        let keep = Set(generations.map { $0.imageFilename ?? Self.imageFilename(for: $0.id) })
        try removeOrphanImages(keeping: keep)
    }

    func delete(_ generation: Generation, remaining: [Generation]) throws {
        let filename = generation.imageFilename ?? Self.imageFilename(for: generation.id)
        let imageURL = imagesDir.appending(component: filename)
        if fm.fileExists(atPath: imageURL.path) {
            try fm.removeItem(at: imageURL)
        }
        try save(remaining)
    }

    private func prepareDirectories() throws {
        try fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)
    }

    private func cleanupTemporaryFiles() throws {
        guard let walker = fm.enumerator(at: root, includingPropertiesForKeys: nil) else { return }
        for case let url as URL in walker where url.lastPathComponent.hasSuffix(".tmp") {
            try? fm.removeItem(at: url)
        }
    }

    private func decodeManifest() throws -> Manifest {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(Manifest.self, from: Data(contentsOf: manifestURL))
    }

    private func writeManifest(_ manifest: Manifest) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func persist(_ generation: Generation) throws -> Record {
        let filename = generation.imageFilename ?? Self.imageFilename(for: generation.id)
        let imageURL = imagesDir.appending(component: filename)
        if !fm.fileExists(atPath: imageURL.path) {
            guard let data = ImageCodec.pngData(generation.image) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try data.write(to: imageURL, options: .atomic)
        }
        return Record(id: generation.id, imageFilename: filename, prompt: generation.prompt,
                      modelID: generation.modelID, modelName: generation.modelName,
                      size: generation.size, steps: generation.steps, seed: generation.seed,
                      duration: generation.duration, settings: generation.settings,
                      date: generation.date)
    }

    private func loadGeneration(from record: Record) -> Generation? {
        let url = imagesDir.appending(component: record.imageFilename)
        guard let image = ImageCodec.cgImage(from: url) else { return nil }
        return Generation(id: record.id, image: image, prompt: record.prompt, modelID: record.modelID,
                          modelName: record.modelName, size: record.size, steps: record.steps,
                          seed: record.seed, duration: record.duration, settings: record.settings,
                          date: record.date, imageFilename: record.imageFilename)
    }

    private func removeOrphanImages(keeping filenames: Set<String>) throws {
        guard let urls = try? fm.contentsOfDirectory(at: imagesDir, includingPropertiesForKeys: nil) else { return }
        for url in urls where url.pathExtension.lowercased() == "png" && !filenames.contains(url.lastPathComponent) {
            try? fm.removeItem(at: url)
        }
    }

    private static func imageFilename(for id: UUID) -> String { "\(id.uuidString).png" }
}

enum ImageCodec {
    /// PNG bytes with no embedded metadata. Used for the Library's durable copy on disk — keep this
    /// signature stable; `LibraryStore` depends on it.
    static func pngData(_ cg: CGImage) -> Data? {
        encode(cg, properties: nil)
    }

    /// XMP namespace + prefix under which the a1111-style `parameters` block is written. A custom
    /// namespace is the only path ImageIO actually persists for arbitrary keys — PNG's text dictionary
    /// only round-trips its known `kCGImagePropertyPNG*` keys, so a bare "parameters" dict entry is
    /// silently dropped (verified). XMP serializes into the PNG as an iTXt packet.
    private static let metadataNamespace = "http://mobilediffuser.app/ns/1.0/"
    private static let metadataPrefix = "md"

    /// PNG bytes that embed provenance, so an exported/shared image carries its prompt, seed, model,
    /// and size/steps. Two channels are written, both verified to round-trip via `CGImageSource`:
    ///   • `Description` (PNG dictionary) — a compact "prompt | seed: N | model | 512px/4steps" line,
    ///     surfaced by Preview, Finder "Get Info", `sips`, and most image tools (also as `dc:description`).
    ///   • `md:parameters` (XMP) — an a1111-style "prompt\nSteps: N, Seed: N, Model: X, Size: WxH" block
    ///     so other tooling can recover the generation settings.
    static func pngData(for gen: Generation) -> Data? {
        let recipe = "\(gen.size)px/\(gen.steps)steps"
        let description = "\(gen.prompt) | seed: \(gen.seed) | \(gen.modelName) | \(recipe)"
        let parameters = """
        \(gen.prompt)
        Steps: \(gen.steps), Seed: \(gen.seed), Model: \(gen.modelName), Size: \(gen.size)x\(gen.size)
        """
        let png: [CFString: Any] = [kCGImagePropertyPNGDescription: description]
        let properties: [CFString: Any] = [kCGImagePropertyPNGDictionary: png]
        return encode(gen.image, properties: properties as CFDictionary, parameters: parameters)
    }

    private static func encode(_ cg: CGImage, properties: CFDictionary?, parameters: String? = nil) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            return nil
        }
        if let parameters, let metadata = parametersMetadata(parameters) {
            CGImageDestinationAddImageAndMetadata(dest, cg, metadata, properties)
        } else {
            CGImageDestinationAddImage(dest, cg, properties)
        }
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    /// Build a `CGImageMetadata` carrying `md:parameters` under our custom XMP namespace.
    private static func parametersMetadata(_ parameters: String) -> CGImageMetadata? {
        let metadata = CGImageMetadataCreateMutable()
        guard CGImageMetadataRegisterNamespaceForPrefix(
                metadata, metadataNamespace as CFString, metadataPrefix as CFString, nil),
              let tag = CGImageMetadataTagCreate(
                metadataNamespace as CFString, metadataPrefix as CFString,
                "parameters" as CFString, .string, parameters as CFString),
              CGImageMetadataSetTagWithPath(metadata, nil, "\(metadataPrefix):parameters" as CFString, tag)
        else { return nil }
        return metadata
    }

    static func cgImage(from url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}
