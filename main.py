import os
import discord
import matplotlib.pyplot as plt
from discord.ext import commands
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

TOKEN = os.environ.get('DISCORD_BOT_TOKEN')
ANALYZE_CHANNEL_ID = os.environ.get('ANALYZE_CHANNEL_ID')
intents=discord.Intents.all()
client = commands.Bot(command_prefix='!',intents=intents)
sia = SentimentIntensityAnalyzer()

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    if message.channel.id == int(ANALYZE_CHANNEL_ID):
        score = sia.polarity_scores(message.content)['compound']
        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S")
        with open("sentiment_scores.txt", "a") as f:
            f.write(f"{timestamp},{score}\n")
    if message.content.startswith("!graph"):
        if message.channel.id == int(os.getenv("GRAPH_CHANNEL_ID")):
            plot_sentiment_scores()
            with open("sentiment_scores.png", "rb") as f:
                await message.channel.send(file=discord.File(f))
        else:
            await message.channel.send("Sorry, the `!graph` command can only be used in the designated channel.")

def plot_sentiment_scores():
    timestamps = []
    scores = []
    with open("sentiment_scores.txt") as f:
        for line in f:
            timestamp, score = line.strip().split(',')
            timestamps.append(timestamp)
            scores.append(float(score))
    plt.plot(range(len(scores)), scores)
    plt.xlabel('Message Index')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis of the Channel')
    plt.tight_layout()

    
    filename = "sentiment_scores.png"
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)


client.run(TOKEN)
