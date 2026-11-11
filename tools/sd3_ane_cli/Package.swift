// swift-tools-version: 5.9
// Mac CLI that runs the app's Core ML SD3 resource folder (split MMDiT, tiled VAE for 1024)
// through the vendored ml-stable-diffusion pipeline. Used to validate resources before device.
import PackageDescription

let package = Package(
    name: "sd3-ane-cli",
    platforms: [.macOS(.v14)],
    dependencies: [.package(path: "../../ml-stable-diffusion")],
    targets: [
        .executableTarget(
            name: "sd3-ane-cli",
            dependencies: [.product(name: "StableDiffusion", package: "ml-stable-diffusion")]
        ),
    ]
)
