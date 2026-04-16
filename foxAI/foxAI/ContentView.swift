//
//  ContentView.swift
//  foxAI
//
//  Created by Fox on 16.04.2026.
//

import SwiftUI

struct ContentView: View {
    @Binding var document: foxAIDocument

    var body: some View {
        TextEditor(text: $document.text)
    }
}

#Preview {
    ContentView(document: .constant(foxAIDocument()))
}
