"""Apply the strixllama management pages to a Jan v0.8.4 source tree.

    python integrations/jan/apply.py [path to a Jan checkout] [--keep-data-dir]

Every edit is anchored against upstream Jan text and fails loudly rather than guessing if an anchor
has moved, so a Jan version this was not written for is a clear error and not a half-applied tree.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The name people see, and the one machines use. The repository, the package, the binary, the
# provider id and the data directory are the slug; the window, the sidebar, the credits and the
# installer show the name.
NAME = 'Strix Llama'
SLUG = 'strixllama'
# Ours, not Jan's: the installer's file name, the uninstall entry and Settings › General show it.
VERSION = '0.1.2'
ARGS = [a for a in sys.argv[1:] if not a.startswith('-')]
KEEP_DATA_DIR = '--keep-data-dir' in sys.argv
JAN = Path(ARGS[0]).resolve() if ARGS else ROOT / 'src/jan'

def replace_once(path, old, new):
    text = path.read_text(encoding='utf-8')
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f'Upstream source changed: {path}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8', newline='\n')

def main():
    shutil.copyfile(HERE / 'strixllama.rs', JAN / 'src-tauri/src/strixllama.rs')
    ui = JAN / 'web-app/src/components/strixllama'
    ui.mkdir(parents=True, exist_ok=True)
    for name in ('StrixLlamaPage.tsx', 'StrixLlamaSync.tsx', 'status.ts', 'strixllama.css'):
        shutil.copyfile(HERE / name, ui / name)
    # StrixLlamaSync polls the manager for the whole app and registers the one provider Jan's chat
    # needs, so a fresh install can chat without first naming an endpoint in a dialog.
    root = JAN / 'web-app/src/routes/__root.tsx'
    replace_once(root, "import { DataProvider } from '@/providers/DataProvider'\n",
                 "import { DataProvider } from '@/providers/DataProvider'\n"
                 "import { StrixLlamaSync } from '@/components/strixllama/StrixLlamaSync'\n")
    replace_once(root, "            <DataProvider />\n", "            <DataProvider />\n            <StrixLlamaSync />\n")
    # Jan's i18n discovers namespaces with import.meta.glob over locales/**/*.json, so dropping the
    # files in is enough - no registration to patch. A language Jan has but we do not falls back to
    # its own fallbackLng, which is en.
    for locale in sorted(p.name for p in (HERE / 'locales').iterdir() if p.is_dir()):
        target = JAN / 'web-app/src/locales' / locale
        if not target.is_dir():
            raise RuntimeError(f'Jan has no locale {locale}; update integrations/jan/locales')
        shutil.copyfile(HERE / 'locales' / locale / 'strixllama.json', target / 'strixllama.json')
    routes = JAN / 'web-app/src/routes/strixllama'
    routes.mkdir(parents=True, exist_ok=True)
    for view in ('models', 'configuration', 'developer'):
        (routes / f'{view}.tsx').write_text(
            "import { createFileRoute } from '@tanstack/react-router'\n"
            "import StrixLlamaPage from '@/components/strixllama/StrixLlamaPage'\n"
            f"export const Route = createFileRoute('/strixllama/{view}')({{\n"
            f"  component: () => <StrixLlamaPage view=\"{view}\" />,\n}})\n", encoding='utf-8')
    lib = JAN / 'src-tauri/src/lib.rs'
    replace_once(lib, 'pub mod core;', 'pub mod core;\nmod strixllama;')
    replace_once(lib, 'tauri::generate_handler![', 'tauri::generate_handler![\n            strixllama::strixllama_request,')
    replace_once(JAN / 'src-tauri/src/main.rs', '#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]', '#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]')
    # The chat goes through the app's Rust HTTP client (reqwest), which picks up the Windows
    # system proxy but not its "bypass for 127.*" list - so with Clash or similar on, every
    # request to the local server was handed to the proxy, and when the proxy could not reach it
    # the chat failed with "Bad Gateway" (seen in app.log: proxy(http://127.0.0.1:7897/) intercepts
    # 'http://127.0.0.1:8080/'). reqwest does honour NO_PROXY, for the registry proxy as well.
    replace_once(JAN / 'src-tauri/src/main.rs', "    app_lib::run();\n", """    // strixllama: the model server is on loopback and must never be routed through a proxy
    let mut no_proxy = String::from("127.0.0.1,localhost,::1");
    if let Ok(existing) = std::env::var("NO_PROXY") {
        if !existing.is_empty() {
            no_proxy.push(',');
            no_proxy.push_str(&existing);
        }
    }
    std::env::set_var("NO_PROXY", &no_proxy);

    app_lib::run();
""")
    nav = JAN / 'web-app/src/components/left-sidebar/NavMain.tsx'
    replace_once(nav, "import { LucideIcon } from 'lucide-react'", "import { LucideIcon, Database, SlidersHorizontal, Terminal } from 'lucide-react'")
    anchor = "  {\n    title: 'common:settings',"
    # Jan renders t(item.title) and its own entries are namespaced keys, so these follow the
    # language Jan is set to instead of being pinned to one, as they were.
    addition = """  { title: 'strixllama:tabs.models', url: '/strixllama/models', icon: Database },
  { title: 'strixllama:tabs.configuration', url: '/strixllama/configuration', icon: SlidersHorizontal },
  { title: 'strixllama:tabs.developer', url: '/strixllama/developer', icon: Terminal },
"""
    replace_once(nav, anchor, addition + anchor)
    drop_nav_entries(nav)
    converge_settings()
    drop_integrations()
    brand(KEEP_DATA_DIR)
    brand_text()
    # Jan's own llama.cpp engine is not loaded at all. Left in, its extension downloads a Vulkan
    # backend on first start (llamacpp-b9967-win-vulkan..., seen in the download tray), starts an
    # embedding server, and registers the provider the picker then has to filter out. Everything
    # that looks it up does so by name with a fallback, as on the platforms where it is absent.
    replace_once(JAN / 'web-app/src/services/core/bundled-extensions.ts', """  {
    load: () => import('@janhq/llamacpp-extension'),
    name: '@janhq/llamacpp-extension',
    productName: 'llama.cpp Inference Engine',
    version: '1.0.1',
    description: 'This extension enables llama.cpp chat completion API calls',
  },
""", """  // strixllama: no llama.cpp engine of Jan's - inference is the server tools/manager.py starts
""")
    print('strixllama: native command, pages, sidebar, settings and branding applied to %s' % JAN)


def cut_lines(path, first_line, last_line, expect_first, expect_last):
    """Delete an inclusive 1-based line range, refusing unless both anchors still match.

    apply.py is run again by every build, so this has to be idempotent: once the range is gone the
    file is shorter and the anchors no longer line up, and the call becomes a no-op.
    """
    lines = path.read_text(encoding='utf-8').split('\n')
    if last_line > len(lines):
        return False                       # already cut
    if expect_first not in lines[first_line - 1] or expect_last not in lines[last_line - 1]:
        return False                       # already cut, or upstream moved
    del lines[first_line - 1:last_line]
    path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return True


def cut_block(path, start_marker):
    """Delete the statement beginning at start_marker, through its matching closing brace.

    Anchor-based rather than by line number, because each cut shifts everything after it.
    """
    text = path.read_text(encoding='utf-8')
    if start_marker not in text:
        return False                       # already cut
    start = text.index(start_marker)
    depth, i, seen = 0, start, False
    while i < len(text):
        if text[i] == '{':
            depth += 1
            seen = True
        elif text[i] == '}':
            depth -= 1
            if seen and depth == 0:
                break
        i += 1
    end = text.index('\n', i) + 1          # take the rest of the closing line, e.g. "}, [])"
    path.write_text(text[:start] + text[end:], encoding='utf-8', newline='\n')
    return True


def drop_declarations(path, names):
    """Delete named import specifiers and whole single-line declarations.

    The web app builds with noUnusedLocals, so every card and menu entry removed here takes its
    icon import - and sometimes a constant - down with it.
    """
    lines = path.read_text(encoding='utf-8').split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        # a specifier on its own line inside a multi-line import
        if any(stripped in (f'{n},', n) for n in names):
            continue
        # a whole single-line import or const
        if any(stripped.startswith(f'import {{ {n} }}') or stripped.startswith(f'const {n} =')
               for n in names):
            continue
        out.append(line)
    path.write_text('\n'.join(out), encoding='utf-8', newline='\n')


def brand(keep_data_dir=False):
    """Make the built app strixllama's rather than Jan's: name, window title, icon, data directory.

    Jan keeps its data in %APPDATA%/<Cargo package name>/data - the threads, the settings, the
    providers - and names the binary after the package too, so the package is renamed along with
    the identifier. That gives this build its own directory instead of sharing an installed Jan's,
    which is what you want for a separate product: two apps writing one settings directory is how
    you lose a conversation history. It does mean an existing Jan install's data is not carried
    over. --keep-data-dir leaves both the package name and the identifier alone.

    This renames a *build* of Jan, which Apache-2.0 allows. NOTICE.md states what it is; do not
    imply that Jan endorses it.
    """
    icons = HERE / 'icons'
    if not (icons / 'icon.ico').is_file():
        subprocess.run([sys.executable, str(HERE / 'make_icons.py')], check=True)
    # A clean Jan checkout ships only icon.png; the other sizes are generated at build time. We
    # write the whole set, because tauri.conf.json's bundle.icon names several of them explicitly
    # and a missing one fails the bundle rather than falling back.
    dst = JAN / 'src-tauri/icons'
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(icons.iterdir()):
        if f.suffix in ('.png', '.ico', '.icns'):
            shutil.copyfile(f, dst / f.name)
            copied += 1

    conf = JAN / 'src-tauri/tauri.conf.json'
    data = json.loads(conf.read_text(encoding='utf-8'))
    data['productName'] = NAME
    data['mainBinaryName'] = SLUG      # strixllama.exe, whatever the product is called
    data['version'] = VERSION
    # Settings › General reads the web app's package version
    web_pkg = JAN / 'web-app/package.json'
    web = json.loads(web_pkg.read_text(encoding='utf-8'))
    if web.get('version') != VERSION:
        web['version'] = VERSION
        web_pkg.write_text(json.dumps(web, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    if not keep_data_dir:
        data['identifier'] = 'dev.rulith.strixllama'
        cargo = JAN / 'src-tauri/Cargo.toml'
        replace_once(cargo, '[package]\nname = "Jan"\n', '[package]\nname = "strixllama"\n')
        replace_once(cargo, 'default-run = "Jan"\n', 'default-run = "strixllama"\n')
        # No jan-cli: it serves Jan's engine, brand() already stops installing it, and Tauri
        # bundles every [[bin]] of the package - so an unbuilt one fails the bundle outright.
        cli_bin = '[[bin]]\nname = "jan-cli"\npath = "src/bin/jan-cli.rs"\nrequired-features = ["cli"]\n'
        text = cargo.read_text(encoding='utf-8')
        if cli_bin in text:
            cargo.write_text(text.replace(cli_bin, '', 1), encoding='utf-8', newline='\n')
        # ...and without the explicit target cargo would auto-discover src/bin/jan-cli.rs and try
        # to compile it without its feature. Discovery off covers src/main.rs too, so the one
        # binary is declared explicitly.
        replace_once(cargo, '[package]\nname = "strixllama"\n', '[package]\nname = "strixllama"\nautobins = false\n')
        replace_once(cargo, '[lib]\nname = "app_lib"\n',
                     '[[bin]]\nname = "strixllama"\npath = "src/main.rs"\n\n[lib]\nname = "app_lib"\n')
        # The Tauri CLI has its own discovery too: it bundles every file under src/bin whatever
        # Cargo.toml says, and fails when the binary was never built. The source goes.
        cli_src = JAN / 'src-tauri/src/bin/jan-cli.rs'
        if cli_src.is_file():
            cli_src.unlink()
            if not any(cli_src.parent.iterdir()):
                cli_src.parent.rmdir()
        # The bundle-identifier constant is only used to look for a legacy settings file to
        # migrate, and the migration deletes the file it copies. Pointed at Jan's directory, a
        # first run would carry off - and remove - an installed Jan's settings.
        constants = JAN / 'src-tauri/src/core/app/constants.rs'
        replace_once(constants, 'pub const TAURI_BUNDLE_IDENTIFIER: &str = "jan.ai.app";',
                     'pub const TAURI_BUNDLE_IDENTIFIER: &str = "dev.rulith.strixllama";')
        replace_once(constants, 'assert_eq!(TAURI_BUNDLE_IDENTIFIER, "jan.ai.app");',
                     'assert_eq!(TAURI_BUNDLE_IDENTIFIER, "dev.rulith.strixllama");')
    # Disable the updater. It points at Jan's endpoints AND carries Jan's signing key, so an
    # upstream release would validate and install — replacing this build with stock Jan, runtime
    # and management pages gone. Jan's own release notes feed goes with it, for the same reason:
    # it would advertise versions that have nothing to do with what is installed.
    plugins = data.get('plugins') or {}
    if 'updater' in plugins:
        plugins.pop('updater')
        data['plugins'] = plugins
    bundle = data.get('bundle') or {}
    if bundle.get('createUpdaterArtifacts'):
        bundle['createUpdaterArtifacts'] = False
        data['bundle'] = bundle
    # The runtime bundle, when one has been made (tools/make_runtime_bundle.py): the server, the
    # ROCm DLLs it needs, the manager and an embedded Python, installed under <app>/runtime so a
    # user needs nothing but the model files. Without one, the build is a development build that
    # runs the repository it was compiled in.
    runtime = ROOT / 'dist' / 'runtime'
    resources = bundle.get('resources') or []
    if isinstance(resources, list):
        resources = {r: r for r in resources}
    resources = {k: v for k, v in resources.items() if not v.startswith('runtime/') and 'jan-cli' not in k}
    staged = JAN / 'src-tauri' / 'runtime'
    if (runtime / 'BUNDLE.json').is_file():
        # Copied into the Tauri project and mapped as a directory: Tauri walks a directory into
        # the target preserving its structure (a glob flattens every match by file name), and a
        # path that climbs out of the project was silently left out of the bundle.
        if staged.is_dir():
            shutil.rmtree(staged)
        shutil.copytree(runtime, staged)
        resources['runtime'] = 'runtime/'
        print('  brand: runtime bundle from %s' % runtime)
    bundle['resources'] = resources
    data['bundle'] = bundle
    # ...and the plugin that reads it: the call is `?`-propagated inside setup(), so an updater
    # with no configuration would stop the application from starting at all.
    replace_once(JAN / 'src-tauri/src/lib.rs',
                 """            #[cfg(not(any(target_os = "ios", target_os = "android")))]
            app.handle()
                .plugin(tauri_plugin_updater::Builder::new().build())?;""",
                 """            // strixllama: no updater. It was configured with Jan's endpoints and Jan's signing
            // key, so an upstream release would verify and install over this build.""")
    conf.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

    # No `jan` command on the user's PATH. Jan copies its CLI into resources/bin at every launch
    # and appends that directory to the Windows user PATH; the CLI serves Jan's engine, which this
    # build never loads, and an application editing the user's environment on startup is not
    # something to inherit. The settings card that offered it went in converge_settings().
    replace_once(JAN / 'src-tauri/src/lib.rs',
                 "            setup::setup_jan_cli(app.handle().clone(), stored_version != app_version);\n",
                 "            // strixllama: no `jan` CLI install - it serves Jan's engine and edits the user's PATH\n"
                 "            let _ = (&stored_version, &app_version);\n")

    # The window title lives in the per-platform config, not in index.html and not in the main one -
    # this is the name in the title bar, which is the first thing anyone sees.
    for name in ('tauri.windows.conf.json', 'tauri.macos.conf.json', 'tauri.linux.conf.json'):
        platform = JAN / 'src-tauri' / name
        if not platform.is_file():
            continue
        pdata = json.loads(platform.read_text(encoding='utf-8'))
        windows = pdata.get('app', {}).get('windows') or []
        changed = False
        for w in windows:
            if w.get('title') == 'Jan':
                w['title'] = NAME
                changed = True
        # One installer. Jan also builds an MSI, which needs the WiX toolset fetched from GitHub
        # at bundle time and adds nothing the NSIS setup does not already do.
        targets = pdata.get('bundle', {}).get('targets')
        if isinstance(targets, list) and 'msi' in targets:
            pdata['bundle']['targets'] = [t for t in targets if t != 'msi']
            changed = True
        # The platform file's own bundle.resources replaces the main one wholesale, so the runtime
        # bundle has to be declared here too or the installer quietly ships without it.
        presources = pdata.get('bundle', {}).get('resources')
        if presources is not None:
            if isinstance(presources, list):
                presources = {r: r for r in presources}
            presources = {k: v for k, v in presources.items() if v != 'runtime/' and 'jan-cli' not in k}
            if 'runtime' in resources:
                presources['runtime'] = 'runtime/'
            if presources != pdata['bundle'].get('resources'):
                pdata['bundle']['resources'] = presources
                changed = True
        if not changed:
            continue
        platform.write_text(json.dumps(pdata, indent=2, ensure_ascii=False) + '\n',
                            encoding='utf-8', newline='\n')

    # build:tauri starts with `tauri icon`, which regenerates every size from icon.png by plain
    # downscaling — including the 16 px one, where make_icons.py deliberately drops the ring and
    # grows the eyes. We ship the whole set already, so drop that step and keep the hinted sizes.
    pkg = JAN / 'package.json'
    scripts = json.loads(pkg.read_text(encoding='utf-8'))
    tauri = scripts['scripts'].get('build:tauri', '')
    if 'build:icon &&' in tauri:
        scripts['scripts']['build:tauri'] = tauri.replace('yarn build:icon && ', '', 1)
        pkg.write_text(json.dumps(scripts, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

    html = JAN / 'web-app/index.html'
    replace_once(html, '<title>Jan</title>', f'<title>{NAME}</title>')
    # The splash: index.html shows /images/jan-logo.png (the waving hand) until the app mounts,
    # and two other places use the same file. One image replaces all three.
    shutil.copyfile(icons / 'icon.png', JAN / 'web-app/public/images/jan-logo.png')
    replace_once(html, '<img src="/images/jan-logo.png" alt="Jan Logo" data-tauri-drag-region />',
                 f'<img src="/images/jan-logo.png" alt="{NAME}" data-tauri-drag-region />')
    replace_once(html, 'Booting up Jan…', f'Starting {NAME}…')
    replace_once(html, '        animation: wave 2s ease-in-out 2.5s infinite;\n',
                 '        animation: none;   /* strixllama: an owl does not wave */\n')
    # The name at the top of the sidebar - the native title bar is hidden behind Jan's own window
    # chrome, so this is the name people actually see.
    sidebar = JAN / 'web-app/src/components/left-sidebar/index.tsx'
    replace_once(sidebar, '<span className="ml-2 font-medium font-studio">Jan</span>',
                 f'<span className="ml-2 font-medium font-studio">{NAME}</span>')
    replace_once(sidebar, '<span className="mr-2 font-medium font-studio">Jan</span>',
                 f'<span className="mr-2 font-medium font-studio">{NAME}</span>')
    # The download tray beside it managed Hub models and engine backends, neither of which this
    # build fetches; models come from the catalog on disk.
    replace_once(sidebar, "              {isLeftPanelOpen && <DownloadManagement />}\n",
                 "              {/* strixllama: no download tray - nothing here is downloaded */}\n")
    text = sidebar.read_text(encoding='utf-8')
    text = text.replace("import { DownloadManagement } from '@/containers/DownloadManegement'\n", "", 1)
    if text.count('isLeftPanelOpen') == 1:   # only its declaration is left, and the app builds with noUnusedLocals
        text = (text.replace("  const { open: isLeftPanelOpen } = useLeftPanel()\n", "", 1)
                    .replace("import { useLeftPanel } from '@/hooks/useLeftPanel'\n", "", 1))
    sidebar.write_text(text, encoding='utf-8', newline='\n')
    # Jan capitalises a provider it has no title for: give ours its name.
    replace_once(JAN / 'web-app/src/lib/utils.ts', "    case 'llamacpp':\n      return 'Llama.cpp'\n",
                 f"    case '{SLUG}':\n      return '{NAME}'\n    case 'llamacpp':\n      return 'Llama.cpp'\n")
    # Two cards Jan shows a fresh install: "download Jan V3.5 for your device" fetches a model for
    # Jan's engine, which this build never loads, and the analytics consent asks about telemetry
    # that is not configured (no PostHog key) and would go to Jan's project if it were.
    root = JAN / 'web-app/src/routes/__root.tsx'
    for line in ("import { useAnalytic } from '@/hooks/useAnalytic'\n",
                 "import { PromptAnalytic } from '@/containers/analytics/PromptAnalytic'\n",
                 "import { useJanModelPrompt } from '@/hooks/useJanModelPrompt'\n",
                 "import { PromptJanModel } from '@/containers/PromptJanModel'\n",
                 "  const { productAnalyticPrompt } = useAnalytic()\n",
                 "  const { showJanModelPrompt } = useJanModelPrompt()\n",
                 "        {productAnalyticPrompt && <PromptAnalytic />}\n",
                 "        {showJanModelPrompt && <PromptJanModel />}\n"):
        text = root.read_text(encoding='utf-8')
        if line in text:
            root.write_text(text.replace(line, '', 1), encoding='utf-8', newline='\n')
    print('  brand: %d icons, productName=%s, binary=%s, identifier=%s' % (copied, NAME, SLUG, data['identifier']))


import re

# The product name wherever a locale string names the product. Not \b: Japanese and Chinese run
# straight into the word, and \w counts their characters as word characters.
PRODUCT_WORD = re.compile(r'(?<![A-Za-z])Jan(?![A-Za-z])')
CREDITS = {
    'en': (f"{NAME} is a build of Jan by Menlo Research (Apache-2.0), with its own inference "
           "runtime and management pages in place of Jan's engines and providers.",
           "It runs on a pwilkin branch of llama.cpp, TheRock ROCm and Tauri. NOTICE.md in the "
           "repository lists every licence."),
    'zh-CN': (f"{NAME} 基于 Menlo Research 的 Jan（Apache-2.0）构建，用自己的推理运行时和管理页面"
              "取代了 Jan 的引擎与模型提供商。",
              "底层依赖 llama.cpp 的 pwilkin 分支、TheRock ROCm 与 Tauri。完整许可见仓库中的 NOTICE.md。"),
}


def brand_text():
    """The name where the app says it: the default assistant, the credits, every locale string.

    Attribution is the one thing not renamed. The credits say what this is built on rather than
    claiming Jan's sentence about its own team, and the other strings that describe Jan itself
    (documentation, release notes, GitHub) belong to cards converge_settings() already removed.
    """
    # The default assistant: seeded by the assistant extension on first run, and the web app's
    # own fallback when no extension answers. Both carry the name and a sentence about Jan.
    sentence = re.compile(r"Jan is a helpful desktop assistant that can reason through complex tasks "
                          r"and use tools to complete them on the user.s behalf\.")
    # a curly apostrophe survives both the single- and the double-quoted string it lands in
    description = ("A local assistant that reasons through complex tasks and uses tools to "
                   "complete them on the user’s behalf.")
    for path in (JAN / 'extensions/assistant-extension/src/index.ts',
                 JAN / 'web-app/src/hooks/useAssistant.ts'):
        text = path.read_text(encoding='utf-8')
        new = sentence.sub(description, text.replace("name: 'Jan',", f"name: '{NAME}',", 1)
                           .replace("avatar: '👋',", "avatar: '🦉',", 1))
        if new != text:
            path.write_text(new, encoding='utf-8', newline='\n')
    # ...and an assistant.json a Jan build wrote before the rename still says Jan: rename it as
    # it is read, in the store, so an existing data directory shows the same name as a new one.
    store = JAN / 'web-app/src/hooks/useAssistant.ts'
    replace_once(store, """  setAssistants: (assistants) => {
    if (assistants) {
      assistants.forEach((a) => (a.id = a.id?.toString())) // new String("id") !== "id"
""", """  setAssistants: (assistants) => {
    if (assistants) {
      assistants.forEach((a) => (a.id = a.id?.toString())) // new String("id") !== "id"
      // strixllama: an assistant written by a Jan build keeps Jan's name on disk
      assistants.forEach((a) => {
        if (a.id === 'jan' && a.name === 'Jan') {
          a.name = '""" + NAME + """'
          a.description = '""" + description + """'
        }
      })
""")

    general = JAN / 'web-app/src/routes/settings/general.tsx'
    # No updater in this build (see brand()), so no "check for updates" either.
    replace_once(general, "              {!AUTO_UPDATER_DISABLED && (",
                 "              {/* strixllama: no updater in this build, see brand() in apply.py */}\n"
                 "              {false && (")
    # Telemetry: there is no key, so nothing is collected, and the consent card would be asking
    # on Jan's behalf. Gated on a constant rather than cut, for the same reason as SHOW_PROVIDERS.
    privacy = JAN / 'web-app/src/routes/settings/privacy.tsx'
    replace_once(privacy, "  return (\n", "  const SHOW_ANALYTICS = false   // strixllama: no telemetry is configured\n  return (\n")
    card = """            <Card
              header={
                <div className="flex items-center justify-between mb-4">
                  <h1 className="font-medium text-foreground text-base">
                    {t('settings:privacy.analytics')}"""
    text = privacy.read_text(encoding='utf-8')
    if 'SHOW_ANALYTICS && (' not in text:
        if text.count(card) != 1:
            raise RuntimeError(f'Upstream source changed: {privacy}')
        start = text.index(card)
        end = text.index('            </Card>\n', start) + len('            </Card>\n')
        text = text[:start] + '            {SHOW_ANALYTICS && (\n' + text[start:end] + '            )}\n' + text[end:]
        privacy.write_text(text, encoding='utf-8', newline='\n')

    # Every locale: the product's name in strings, the credits replaced (English and Chinese
    # written here; the others drop the keys and fall back to English, which is Jan's own
    # fallback rule) rather than reworded into a claim about who built Jan.
    def walk(node, locale):
        if isinstance(node, dict):
            for key in list(node):
                if key in ('creditsDesc1', 'creditsDesc2'):
                    if locale in CREDITS:
                        node[key] = CREDITS[locale][int(key[-1]) - 1]
                    else:
                        del node[key]
                else:
                    node[key] = walk(node[key], locale)
            return node
        if isinstance(node, list):
            return [walk(x, locale) for x in node]
        if isinstance(node, str):
            return PRODUCT_WORD.sub(NAME, node)
        return node
    for path in sorted((JAN / 'web-app/src/locales').glob('*/*.json')):
        if path.name == 'strixllama.json':
            continue
        original = path.read_text(encoding='utf-8')
        data = walk(json.loads(original), path.parent.name)
        text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        if json.loads(text) != json.loads(original):
            path.write_text(text, encoding='utf-8', newline='\n')


def converge_settings():
    """Strip the settings surfaces this build has no path through.

    Everything here is either dead (it drives Jan's own llama.cpp engine or the Hub, neither of
    which this build ever loads) or actively wrong (the updater would replace a custom binary, and
    the Resources/Community links point at upstream Jan rather than at this fork).
    """
    general = JAN / 'web-app/src/routes/settings/general.tsx'
    # Resources + Community cards: upstream Jan's docs, release notes, GitHub and Discord
    cut_lines(general, 554, 643, '{/* Resources */}', '</Card>')
    # HuggingFace token: only ever used by the Hub downloads that the sidebar no longer exposes
    cut_lines(general, 473, 551, '<CardItem', '/>')
    # Jan CLI install/uninstall: serves models through Jan's engine, which is never loaded here
    cut_lines(general, 409, 439, '{IS_TAURI && (', ')}')
    # and the state, effects and handlers the two removed cards were the only readers of.
    # These are anchor-based: line numbers shift as soon as the first cut lands.
    cut_block(general, "  useEffect(() => {\n    if (!IS_TAURI) return")   # CLI status probe
    cut_block(general, "  const handleInstallCli = async () => {")
    cut_block(general, "  const handleUninstallCli = async () => {")
    drop_declarations(general, ('IconBrandDiscord', 'IconBrandGithub', 'IconExternalLink',
                                'Input', 'TOKEN_VALIDATION_TIMEOUT_MS',
                                'huggingfaceToken', 'setHuggingfaceToken', 'invoke'))
    for decl in ('  const [isValidatingToken, setIsValidatingToken] = useState(false)\n',
                 '  const [cliInstalled, setCliInstalled] = useState<boolean | null>(null)\n',
                 '  const [cliPath, setCliPath] = useState<string | null>(null)\n',
                 '  const [isCliLoading, setIsCliLoading] = useState(false)\n'):
        general.write_text(general.read_text(encoding='utf-8').replace(decl, '', 1),
                           encoding='utf-8', newline='\n')

    # The model picker in the chat header lists every active provider, so Jan's own llama.cpp
    # engine and the remote APIs (Anthropic, Azure, Gemini, ...) show up there even after the
    # settings section is gone. Filter at the source, so all five uses in that file follow.
    # Exclusions rather than a name allow-list: a provider you configured yourself keeps working
    # whatever you called it.
    picker = JAN / 'web-app/src/containers/DropdownModelProvider.tsx'
    replace_once(picker, """    providers,
    getProviderByName,""", """    providers: allProviders,
    getProviderByName,""")
    unmemoized = """  // strixllama: this build serves one local endpoint from tools/manager.py. Jan's bundled
  // engines never load a model here, and the remote APIs are not what it is for.
  const providers = allProviders.filter(
    (p) =>
      p.provider !== 'llamacpp' &&
      p.provider !== 'mlx' &&
      !predefinedProviders.some((e) => e.provider.includes(p.provider))
  )
"""
    # Memoised, and it matters: the list is a dependency of the effect that selects a thread's
    # model. A fresh array every render re-ran that effect, which set state, which rendered again -
    # React error #185 the moment any existing thread was opened.
    memoized = """  // strixllama: this build serves one local endpoint from tools/manager.py. Jan's bundled
  // engines never load a model here, and the remote APIs are not what it is for. Memoised because
  // the list feeds the effect that selects a thread's model; a new array per render loops it.
  const providers = useMemo(
    () =>
      allProviders.filter(
        (p) =>
          p.provider !== 'llamacpp' &&
          p.provider !== 'mlx' &&
          !predefinedProviders.some((e) => e.provider.includes(p.provider))
      ),
    [allProviders]
  )
"""
    text = picker.read_text(encoding='utf-8')
    if memoized not in text:
        if unmemoized in text:             # a tree an earlier apply.py left with the looping array
            text = text.replace(unmemoized, memoized, 1)
        else:
            anchor = "  const [displayModel, setDisplayModel] = useState<string>('')"
            if text.count(anchor) != 1:
                raise RuntimeError(f'Upstream source changed: {picker}')
            text = text.replace(anchor, memoized + anchor, 1)
        picker.write_text(text, encoding='utf-8', newline='\n')
    # With Jan's engine filtered out, its "first llamacpp model" fallback for a new chat never
    # fires and the picker opens on "select a model". Fall back to the first provider that has any.
    replace_once(picker, """          const llamacppProvider = providers.find(
            (p) => p.provider === 'llamacpp' && p.active && p.models.length > 0
          )""", """          const llamacppProvider = providers.find(
            (p) => p.active && p.models.length > 0 // strixllama: the local provider, not Jan's engine
          )""")
    replace_once(picker, """            selectModelProvider('llamacpp', firstModel.id)
            setLastUsedModel('llamacpp', firstModel.id)""",
                 """            selectModelProvider(llamacppProvider.provider, firstModel.id)
            setLastUsedModel(llamacppProvider.provider, firstModel.id)""")
    # The gear beside the provider opens Jan's provider page: base URL, API keys, a Delete button
    # that would take the only provider with it. For ours it opens the Configuration page instead.
    replace_once(picker, """                            navigate({
                              to: route.settings.providers,
                              params: { providerName: providerInfo.provider },
                            })""", """                            if (providerInfo.provider === 'strixllama') {
                              navigate({ to: '/strixllama/configuration' as '/strixllama/models' })
                            } else {
                              navigate({
                                to: route.settings.providers,
                                params: { providerName: providerInfo.provider },
                              })
                            }""")

    # The provider page is still reachable by URL: keep it from deleting the one provider.
    replace_once(JAN / 'web-app/src/routes/settings/providers/$providerName.tsx',
                 "                <DeleteProvider provider={provider} />\n",
                 "                {provider?.provider !== 'strixllama' && <DeleteProvider provider={provider} />}\n")

    menu = JAN / 'web-app/src/containers/SettingsMenu.tsx'
    text = menu.read_text(encoding='utf-8')
    for entry in ('local_api_server',   # we serve on 8080 from tools/manager.py, not from Jan
                  'https_proxy',        # only mattered for the remote providers dropped below
                  'hardware'):          # GPU detection for Jan's engine
        block = f"""    {{
      title: 'common:{entry}',
      route: route.settings.{entry},"""
        if block not in text:
            continue
        start = text.index(block)
        end = text.index('    },\n', start) + len('    },\n')
        text = text[:start] + text[end:]
    # only the strixllama provider is real here: Jan's llama.cpp never loads a model in this build and
    # the remote providers are not what this fork is for
    old_filter = """  const activeProviders = providers.filter((provider) => {
    if (!provider.active) return false
    if (!IS_MACOS && provider.provider === 'mlx') return false
    return true
  })"""
    new_filter = """  const activeProviders = providers.filter((provider) => {
    if (!provider.active) return false
    // strixllama: this build serves every model from tools/manager.py, so Jan's own engines and the
    // remote APIs have nothing to configure. Their routes still resolve if visited directly.
    return provider.provider.toLowerCase().includes('strixllama')
  })"""
    if old_filter in text:
        text = text.replace(old_filter, new_filter, 1)
        text = text.replace("""  const hiddenProviders = providers.filter((provider) => {
    if (provider.active) return false
    if (!IS_MACOS && provider.provider === 'mlx') return false
    return true
  })""", """  const hiddenProviders: typeof providers = []""", 1)

    # "Model providers" is a concept this build does not have: one local server, started by
    # tools/manager.py, and nothing to choose between. The whole section goes.
    #
    # It is gated on a constant rather than deleted because web-app/tsconfig.app.json sets
    # noUnusedLocals: deleting the only use of createProvider, AddProviderDialog and IconPlus turns
    # three unused declarations into build errors, and chasing those down removes things that are
    # genuinely upstream Jan's. The bundler drops the dead branch anyway.
    gate = """  const SHOW_PROVIDERS = false   // strixllama: one local backend, nothing to choose between
  const activeProviders"""
    if 'const SHOW_PROVIDERS' not in text:
        text = text.replace("  const activeProviders", gate, 1)
    section_open = """          {/* Model Providers section */}
          <div className="mt-4">"""
    section_close = """            </div>
            <div className="m-3" />
          </div>"""
    if section_open in text and text.count(section_close) == 1:
        text = text.replace(section_open, """          {/* Model Providers section - see converge_settings() in integrations/jan/apply.py */}
          {SHOW_PROVIDERS && (
          <div className="mt-4">""", 1)
        text = text.replace(section_close, section_close + """
          )}""", 1)
    elif 'SHOW_PROVIDERS && (' not in text:
        raise SystemExit('apply.py: the Model Providers section in SettingsMenu.tsx has moved')
    menu.write_text(text, encoding='utf-8', newline='\n')
    drop_declarations(menu, ('IconCircles', 'IconCpu', 'IconWorld'))


def drop_nav_entries(nav):
    """Hide the Jan features this build has no path through.

    The sidebar should describe what the app can actually do. Hub downloads models for Jan's own
    engine, which this build never loads - every model comes from the strixllama manager instead - and
    agent chats and projects are Jan surfaces none of the strixllama work touches. The routes stay
    reachable by URL; only the entry points go, leaving chat, search, the three strixllama pages and
    settings.
    """
    text = nav.read_text(encoding='utf-8')
    for title in ("common:newAgentChat", "common:projects.new", "common:hub"):
        marker = f"    title: '{title}',"
        if marker not in text:
            continue                       # already dropped
        start = text.rindex("  {\n", 0, text.index(marker))
        depth, i = 0, start
        while i < len(text):               # walk to the entry's matching brace
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        end = text.index('\n', text.index(',', i)) + 1
        text = text[:start] + text[end:]

    # the entries took their icons and callbacks with them, and the web app builds with
    # noUnusedLocals. The *Handle types are still referenced by the NavMainIcon union, so drop only
    # the icon values from each import.
    for old, new in (
            ("import {\n  FolderPlusIcon,\n  type FolderPlusIconHandle,\n} from '@/components/animated-icon/folder-plus'",
             "import { type FolderPlusIconHandle } from '@/components/animated-icon/folder-plus'"),
            ("import { BlocksIcon, type BlocksIconHandle } from '../animated-icon/blocks'",
             "import { type BlocksIconHandle } from '../animated-icon/blocks'"),
            ("import {\n  BotIcon,\n  type BotIconHandle,\n} from '@/components/animated-icon/bot'",
             "import { type BotIconHandle } from '@/components/animated-icon/bot'")):
        text = text.replace(old, new, 1)
    for param in ("onNewProject", "onJanClaw"):
        text = text.replace(f"  {param}: () => void,\n", f"  _{param}: () => void,\n", 1)
        text = text.replace(f"  {param}: () => void\n", f"  _{param}: () => void\n", 1)
    nav.write_text(text, encoding='utf-8', newline='\n')


def cut_span(path, start_marker, end_marker, keep_end):
    """Delete from start_marker up to end_marker (kept when keep_end), idempotent."""
    text = path.read_text(encoding='utf-8')
    if start_marker not in text:
        return False                       # already cut
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    if not keep_end:
        end += len(end_marker)
    path.write_text(text[:start] + text[end:], encoding='utf-8', newline='\n')
    return True


def drop_integrations():
    """Remove the Integrations section (MCP servers, Claude Code) from the settings menu.

    This build only ever talks to the local strixllama server; the agent integrations are Jan
    surfaces none of the strixllama work touches. The routes stay reachable by URL.
    """
    menu = JAN / 'web-app/src/containers/SettingsMenu.tsx'
    cut_span(menu, "  const integrationSettings = [\n", "  ]\n", keep_end=False)
    # prefix only: converge_settings() rewrites the rest of that comment line, and runs before this
    cut_span(menu, "          {/* Integrations section */}\n", "          {/* Model Providers section", keep_end=True)
    drop_declarations(menu, ('IconTopologyStar3',))


if __name__ == '__main__':
    main()
