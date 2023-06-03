from flask import Flask, render_template, request
import movieR
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        movie = request.form.get('movie')
        if movie:
            data = movieR.recommend(movie)
            return render_template('index.html', datam=data)
    
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True,port=5100, host='0.0.0.0')











# def hello_world():
    
#     movies = request.args.get('movie')  # Get the 'movie' parameter from the query string
#     if movies:
#         data = movieR.recommend(movies)
#         return render_template('index.html', datam=data)
#     else:
#         return "Please provide a movie parameter in the URL."
