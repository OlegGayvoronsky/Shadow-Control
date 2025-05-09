import json
import os
import shutil
from pathlib import Path
import vdf


class CreateGameFolders:
    def __init__(self, config_path="config/steam_folder.json", games_dir="games"):
        self.config_path = Path(__file__).resolve().parent.parent / Path(config_path)
        self.games_dir = Path(__file__).resolve().parent.parent / Path(games_dir)
        self.games_dir.mkdir(exist_ok=True)
        self.steam_path = self._get_steam_path()
        self.steamapps = self.steam_path / "steamapps"
        self.common = self.steamapps / "common"
        self.librarycache = self.steam_path / "appcache" / "librarycache"

    def _get_steam_path(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return Path(json.load(f).get("steam_path"))

    def read_manifest(self, game_folder):
        manifest_path = game_folder / "appmanifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def create_folders_from_steam(self):
        acf_files = list(self.steamapps.glob("appmanifest_*.acf"))
        for file in acf_files:
            try:
                self._process_acf_file(file)
            except Exception as e:
                print(f"Ошибка при обработке {file.name}: {e}")
                if hasattr(self, 'game_folder') and self.game_folder.exists():
                    shutil.rmtree(self.game_folder, ignore_errors=True)

    def _process_acf_file(self, file):
        with open(file, encoding="utf-8") as f:
            data = vdf.load(f)
            state = data.get("AppState", {})
            appid = state.get("appid")
            installdir = state.get("installdir")
            name = state.get("name")

            if not (installdir and name):
                return

            steam_game_path = self.common / installdir
            if not steam_game_path.exists():
                return

            self.game_folder = self.games_dir / installdir

            exe_files = list(steam_game_path.rglob("*.exe"))
            os.makedirs(self.game_folder, exist_ok=True)

            assets_path = self.librarycache / appid
            cover_path = assets_path / "library_600x900.jpg"
            if not cover_path.exists():
                cover_path = Path("")
            hero_path = assets_path / "library_hero.jpg"
            if not hero_path.exists():
                hero_path = Path("")
            logo_path = assets_path / "logo.png"
            if not logo_path.exists():
                logo_path = Path("")

            assets = {
                "cover": str(cover_path).replace("\\", "/"),
                "hero": str(hero_path).replace("\\", "/"),
                "logo": str(logo_path).replace("\\", "/")
            }

            game_data = {
                "appid": appid,
                "name": name,
                "installdir": installdir,
                "exe": [str(e).replace("\\", "/") for e in exe_files],
                "assets": assets
            }

            manifest_path = self.game_folder / "appmanifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(game_data, f, indent=4, ensure_ascii=False)