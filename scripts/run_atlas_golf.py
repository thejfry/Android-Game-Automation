from games.atlas_golf.atlas_golf_bot import AtlasGolfBot

if __name__ == "__main__":
    bot = AtlasGolfBot("games/atlas_golf/config.yaml")
    # bot.start(render_hud=True)
    bot.start_naive_approach()
