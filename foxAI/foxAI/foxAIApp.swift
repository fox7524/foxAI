//
//  foxAIApp.swift
//  foxAI
//
//  Created by Fox on 16.04.2026.
//

import SwiftUI

@main
struct foxAIApp: App {
    var body: some Scene {
        DocumentGroup(newDocument: foxAIDocument()) { file in
            ContentView(document: file.$document)
        }
    }
}
