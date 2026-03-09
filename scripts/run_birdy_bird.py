from games.birdy_bird.bot import BirdyBirdBot

if __name__ == "__main__":
    bot = BirdyBirdBot("games/birdy_bird/config.yaml")
    bot.start(render_hud=True)
