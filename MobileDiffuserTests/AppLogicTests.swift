// Unit tests for the app's pure presentation logic (no SwiftUI / engine instances needed).
//
// TO WIRE THIS UP (one-time, ~2 min in Xcode — safer than hand-editing the .xcodeproj):
//   1. File ▸ New ▸ Target… ▸ Unit Testing Bundle. Name it "MobileDiffuserTests",
//      "Target to be Tested" = MobileDiffuser.
//   2. Delete the auto-generated MobileDiffuserTests.swift; drag THIS file into the new target
//      (check "MobileDiffuserTests" in Target Membership).
//   3. ⌘U runs it. (The scheme's Test action picks the new target up automatically.)
//
// These cover the friendly-error mapping (incl. the recoverable insufficientMemory / pausedForHeat
// cases that pair with the canvas Retry button) and the duration formatter.

import XCTest
import DiffusionCore
@testable import MobileDiffuser

@MainActor
final class AppLogicTests: XCTestCase {

    func testFriendlyErrorMapsRecoverableEngineErrors() {
        XCTAssertTrue(AppModel.friendlyError(EngineError.unsupportedOnDevice).contains("smaller size"))
        XCTAssertTrue(AppModel.friendlyError(EngineError.insufficientMemory).contains("free memory"))
        XCTAssertTrue(AppModel.friendlyError(EngineError.pausedForHeat).contains("cool down"))
        XCTAssertTrue(AppModel.friendlyError(EngineError.decodeFailed).contains("decoding"))
        // invalidRequest passes its message through verbatim.
        XCTAssertEqual(AppModel.friendlyError(EngineError.invalidRequest("steps must be positive")),
                       "steps must be positive")
    }

    func testFriendlyErrorMapsCancellationAndNetwork() {
        XCTAssertEqual(AppModel.friendlyError(CancellationError()), "Cancelled")
        XCTAssertEqual(AppModel.friendlyError(URLError(.notConnectedToInternet)), "No connection")
        XCTAssertEqual(AppModel.friendlyError(URLError(.timedOut)), "Network error")
    }

    func testFriendlyErrorFallsBackForUnknownError() {
        // A non-engine, non-URL error must NOT surface a raw enum/NSError dump — it gets the generic,
        // retry-encouraging message (the whole reason this mapping exists).
        let msg = AppModel.friendlyError(NSError(domain: "x", code: 42))
        XCTAssertTrue(msg.contains("try again"))
        XCTAssertFalse(msg.contains("Error Domain"))
    }

    func testFriendlyDownloadError() {
        XCTAssertEqual(AppModel.friendlyDownloadError(URLError(.notConnectedToInternet)), "No connection")
        XCTAssertEqual(AppModel.friendlyDownloadError(NSError(domain: "x", code: 1)), "Download failed")
    }

    func testFormatDuration() {
        XCTAssertEqual(formatDuration(0.5), "0.5s")
        XCTAssertEqual(formatDuration(12.4), "12.4s")
        XCTAssertEqual(formatDuration(59.9), "59.9s")
        XCTAssertEqual(formatDuration(65), "1m 05s")
        XCTAssertEqual(formatDuration(3661), "61m 01s")
    }
}
