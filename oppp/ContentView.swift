//
//  ContentView.swift
//  oppp
//
//  Created by Deepak lenka on 12/16/24.
//
//
import SwiftUI
import MLXVLM
import MLXLMCommon
import PhotosUI

// Move ModelRegistry to top level
enum ModelRegistry {
    static public let paligemma3bMix448_8bit = ModelConfiguration(
        id: "mlx-community/paligemma-3b-mix-448-8bit",
        defaultPrompt: "Describe the image in English"
    )
}

struct ContentView: View {
    // Your existing properties
    @State private var image: UIImage?
    @State private var result: String = ""
    @State private var isLoading = false
    @State private var imageSelection: PhotosPickerItem?
    @State private var modelContainer: ModelContainer?

    var body: some View {
        VStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .cornerRadius(12)
                    .padding()
                    .frame(height: 300)
            }

            PhotosPicker(selection: $imageSelection, matching: .images) {
                Text("Select Image")
            }
            .onChange(of: imageSelection) { _, newValue in
                Task {
                    if let newValue {
                        if let data = try? await newValue.loadTransferable(type: Data.self), let uiImage = UIImage(data: data) {
                            image = uiImage
                            isLoading = true
                            if let ciImage = CIImage(image: uiImage) {
                                try await processImage(ciImage)
                            }
                            isLoading = false
                        }
                    }
                }
            }

            if isLoading {
                ProgressView()
            } else {
                Text(result)
                    .padding()
            }
        }
        .task {
            do {
                modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: ModelRegistry.paligemma3bMix448_8bit
                ) { progress in
                    debugPrint("Downloading \(ModelRegistry.paligemma3bMix448_8bit.id): \(Int(progress.fractionCompleted * 100))%")
                }
            } catch {
                debugPrint(error)
            }
        }
    }
}

extension ContentView {
    private func processImage(_ ciImage: CIImage) async throws {
        guard let container = modelContainer else { return }
        
        // Reset result before processing new image
        await MainActor.run { result = "" }

        var input = UserInput(prompt: "Describe the image in English", images: [.ciImage(ciImage)])
        input.processing.resize = .init(width: 448, height: 448)

        _ = try await container.perform { [input] context in
            let input = try await context.processor.prepare(input: input)

            return try MLXLMCommon.generate(input: input, parameters: .init(), context: context) { tokens in
                Task { @MainActor in
                    self.result += context.tokenizer.decode(tokens: tokens)
                }

                return tokens.count >= 800 ? .stop : .more
            }
        }
    }
}

// End of file. No additional code.
