// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

import SwiftUI

/// Creative-studio shell: three sections (Create / Library / Settings) as a tab bar on iOS and a
/// sidebar split on macOS, all sharing one `AppModel`. Respects the in-app appearance override
/// (which defaults to following the system scheme). Model management lives in Settings + the Create
/// toolbar, not as a top-level section.
struct ContentView: View {
    @State private var model = AppModel()

    var body: some View {
        shell
            .tint(Theme.accent)
            .background(Theme.bg)
            .preferredColorScheme(model.appearance.colorScheme)
    }

    @ViewBuilder private var shell: some View {
        #if os(macOS)
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Tab.allCases.filter { $0 != .settings }) { tab in
                    SidebarRow(tab: tab, selected: model.tab == tab) {
                        withAnimation(Motion.select) { model.tab = tab }
                    }
                }
                Spacer(minLength: 0)
                // Settings pinned to the bottom of the sidebar, like most desktop apps.
                SidebarRow(tab: .settings, selected: model.tab == .settings) {
                    withAnimation(Motion.select) { model.tab = .settings }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .frame(maxHeight: .infinity, alignment: .top)
            .navigationSplitViewColumnWidth(min: 200, ideal: 216)
        } detail: {
            screen(for: model.tab)
                .frame(minWidth: 560, minHeight: 480)
        }
        #else
        TabView(selection: $model.tab) {
            ForEach(Tab.allCases) { tab in
                NavigationStack { screen(for: tab).navigationTitle(tab.title) }
                    .tabItem { Label(tab.title, systemImage: tab.icon) }
                    .tag(tab)
            }
        }
        #endif
    }

    @ViewBuilder private func screen(for tab: Tab) -> some View {
        switch tab {
        case .create: CreateView(model: model)
        case .library: LibraryView(model: model)
        case .settings: SettingsView(model: model)
        }
    }
}

#if os(macOS)
/// A sidebar item with a violet-tinted selected state, so the navigation matches the studio accent
/// instead of the system-blue list-selection highlight.
private struct SidebarRow: View {
    let tab: Tab
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(tab.title, systemImage: tab.icon)
                .labelStyle(.titleAndIcon)
                .font(.body)
                .foregroundStyle(selected ? Theme.accent : Theme.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(selected ? Theme.accentSoft : Color.clear,
                            in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
#endif

#Preview { ContentView() }
