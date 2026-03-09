from games.atlas_bowling.atlas_bowling_bot import AtlasBowlingBot

if __name__ == "__main__":
    bot = AtlasBowlingBot("games/atlas_bowling/config.yaml")
    bot.start(render_hud=True)
