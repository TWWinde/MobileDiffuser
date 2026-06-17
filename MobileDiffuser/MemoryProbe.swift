import Darwin
import Foundation

/// Lightweight RSS / app-memory probe for on-device profiling.
/// Use it to bracket suspicious code regions and print before/after deltas.
enum MemoryProbe {
    /// Resident set size in bytes (what the kernel charges against the app).
    static func residentBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        // `phys_footprint` is the value Apple's jetsam compares against the
        // app's memory limit — much more accurate than `resident_size`.
        return info.phys_footprint
    }

    static func residentMB() -> Double {
        Double(residentBytes()) / 1_048_576
    }

    /// Best-effort total physical memory of the device.
    static func deviceTotalMB() -> Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576
    }

    /// Bytes the kernel will currently let this process allocate before
    /// triggering a jetsam OOM. This is the *real* memory budget — entitlements
    /// like `com.apple.developer.kernel.increased-memory-limit` only matter if
    /// this number reflects them.
    /// Returns 0 on platforms / OS versions where the API is unavailable.
    static func availableBytes() -> UInt64 {
#if canImport(os)
        if #available(iOS 13.0, macOS 10.15, *) {
            return UInt64(os_proc_available_memory())
        }
#endif
        return 0
    }

    static func availableMB() -> Double {
        Double(availableBytes()) / 1_048_576
    }

    /// Print "[MEM] tag: X MB (device 8192 MB)".
    static func log(_ tag: String) {
        print(String(
            format: "[MEM] %-28@: %7.1f MB used, %7.1f MB headroom  (device %.0f MB)",
            tag as NSString,
            residentMB(),
            availableMB(),
            deviceTotalMB()
        ))
    }
}
