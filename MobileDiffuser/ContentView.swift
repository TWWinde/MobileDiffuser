// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Dark creative-studio shell: four sections (Models / Create / Library / Settings) as a tab bar
/// on iOS and a top tab strip on macOS, all sharing one `AppModel`.
struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        TabView(selection: $model.tab) {
            ForEach(Tab.allCases) { tab in
                screen(for: tab)
                    .tabItem { Label(tab.title, systemImage: tab.icon) }
                    .tag(tab)
            }
        }
        .tint(Theme.accent)
        .preferredColorScheme(.dark)
        .background(Theme.bg)
    }

    @ViewBuilder private func screen(for tab: Tab) -> some View {
        switch tab {
        case .models: ModelsView(model: model)
        case .create: CreateView(model: model)
        case .library: LibraryView(model: model)
        case .settings: SettingsView(model: model)
        }
    }
}

#Preview { ContentView() }
