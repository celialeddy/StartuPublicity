from flask import render_template
from flask import request
from insight_flask_app import app
import pandas as pd
from insight_flask_app.preprocess_text import preprocess_text
from insight_flask_app.calculate_model_prediction import calculate_model_prediction

@app.route('/')
@app.route('/index')
@app.route('/input')
def input():
  return render_template("input.html")

@app.route('/input_nonnumeric')
def input_nonnumeric():
  return render_template("input_nonnumeric.html")

@app.route('/input_notext')
def input_notext():
  return render_template("input_notext.html")

@app.route('/input_example')
def input_example():
  return render_template("input_example.html")

@app.route('/output')
def output():
  title = request.args.get('title')
  article_text = request.args.get('article_text')
  num_images = request.args.get('num_images')
  if not title:
    return render_template("input_notext.html")
  if not article_text:
    return render_template("input_notext.html")
  if not num_images:
    num_images=0
  if not num_images.isnumeric():
    return render_template("input_nonnumeric.html")
  [df, min_claps_topic, max_claps_topic, max_max_topic] = preprocess_text(title,article_text,num_images)
  print(title,num_images)
  [min_claps, max_claps, instruction, max_claps_inst, max_claps_fb, perturb_fb] = calculate_model_prediction(df)
  min_claps = int(min_claps)
  max_claps = int(max_claps)
  if max_claps_inst > 0:
    return render_template("output.html",title=title,article_text=article_text,
      min_claps=min_claps,
      max_claps=max_claps,
      min_claps_topic=min_claps_topic,
      max_claps_topic=max_claps_topic,
      max_max_topic=max_max_topic,
      instruction=instruction,
      max_claps_inst=max_claps_inst,
      max_claps_fb=max_claps_fb,
      perturb_fb=perturb_fb)
  else:
    return render_template("output2.html",title=title,article_text=article_text,
      min_claps=min_claps,
      max_claps=max_claps,
      min_claps_topic=min_claps_topic,
      max_claps_topic=max_claps_topic,
      max_max_topic=max_max_topic,
      max_claps_fb=max_claps_fb,
      perturb_fb=perturb_fb)

@app.route('/about')
def about():
  return render_template("about.html")