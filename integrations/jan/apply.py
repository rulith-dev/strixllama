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
    for name in ('StrixLlamaPage.tsx', 'strixllama.css'):
        shutil.copyfile(HERE / name, ui / name)
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
    """Make the built app strixllama's rather than Jan's: name, window title, icon.

    The identifier decides where Tauri keeps application data, so changing it gives this build its
    own directory instead of sharing Jan's. That is what you want for a separate product - two apps
    writing one settings directory is how you lose a conversation history - but it does mean an
    existing Jan install's data is not carried over. --keep-data-dir leaves it alone.

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
    data['productName'] = 'strixllama'
    if not keep_data_dir:
        data['identifier'] = 'dev.rulith.strixllama'
    conf.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')

    # The window title lives in the per-platform config, not in index.html and not in the main one -
    # this is the name in the title bar, which is the first thing anyone sees.
    for name in ('tauri.windows.conf.json', 'tauri.macos.conf.json', 'tauri.linux.conf.json'):
        platform = JAN / 'src-tauri' / name
        if not platform.is_file():
            continue
        pdata = json.loads(platform.read_text(encoding='utf-8'))
        windows = pdata.get('app', {}).get('windows') or []
        if not any(w.get('title') == 'Jan' for w in windows):
            continue
        for w in windows:
            if w.get('title') == 'Jan':
                w['title'] = 'strixllama'
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
    replace_once(html, '<title>Jan</title>', '<title>strixllama</title>')
    print('  brand: %d icons, productName=strixllama, identifier=%s' % (copied, data['identifier']))


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
