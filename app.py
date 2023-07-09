from flask import Flask, render_template
from analysis.getData import getNextWeekPicks

app = Flask(__name__)

@app.route('/')
def index():
    picks = getNextWeekPicks()
    goalkeepers = filter(lambda player: player["position"] == "Goalkeeper", picks)
    defenders = filter(lambda player: player["position"] == "Defender", picks)
    midfielders = filter(lambda player: player["position"] == "Midfielder", picks)
    forwards = filter(lambda player: player["position"] == "Forward", picks)
    return render_template('home.html', goalkeepers=goalkeepers, defenders=defenders, midfielders=midfielders, forwards=forwards)

# @app.route('/stats')
# def stats():
    # return render_template('stats.html', title='Stats')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=443)
    