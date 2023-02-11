import os
import discord
import matplotlib.pyplot as plt
from discord.ext import commands
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from keep_alive import keep_alive
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

    num_msgs = len(timestamps)
    interval = num_msgs // 5
    if interval == 0:
        interval = 1

    plt.plot(range(len(scores)), scores)
    plt.xlabel('Message Index')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis of the Channel')
    plt.xticks(range(0, num_msgs, interval), range(0, num_msgs, interval), rotation=45)
    plt.tight_layout()

    filename = "sentiment_scores.png"
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)


keep_alive()
client.run(TOKEN)
