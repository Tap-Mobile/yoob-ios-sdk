import SwiftUI
import Yoob

@main
struct QuickStartApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// Asks your backend for a Yoob session. The backend calls the Yoob API with your secret key; the app never sees it.
enum Backend {
    static let tokenURL = URL(string: Bundle.main.object(forInfoDictionaryKey: "YoobTokenURL") as? String ?? "")!

    static func credentials(for character: String) async throws -> YoobCredentials {
        var request = URLRequest(url: tokenURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["character": character])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw YoobError.unauthorized }
        return try JSONDecoder().decode(YoobCredentials.self, from: data)
    }
}
