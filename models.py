# from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import yfinance as yf
db = SQLAlchemy()
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    balance = db.Column(db.Float, default=1000000)  # 1 million test credits
    orders = db.relationship('Order', backref='owner', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)




class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)

class StockDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    change = db.Column(db.Float, nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=False)
    stock= db.relationship('Stock', backref='stock_details', lazy=True)
class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    order_type = db.Column(db.String(4), nullable=False)  # BUY or SELL
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(10), nullable=False)  # PENDING, COMPLETED, etc.
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # user= db.relationship('User', backref='orders', lazy=True, uselist=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=False)
    stock= db.relationship('Stock', backref='orders', lazy=True)


class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user= db.relationship('User', backref='portfolio', lazy=True)

    @property
    def calc_profit(self):
        stock_info= yf.Ticker(self.symbol).info
        invested_amount= self.quantity * self.price
        current_price= stock_info['currentPrice']
        profit_percent= round(100*(current_price - self.price) / self.price,2)

        return profit_percent

    def __repr__(self):
        return f"Portfolio({self.symbol}, {self.quantity}, {self.price})"

class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    url = db.Column(db.String(200), nullable=False)
    date = db.Column(db.Date, nullable=False)
    content= db.Column(db.Text, nullable=False)
    source= db.Column(db.String(100), nullable=False) # e.g. CNN,Reddit,Twitter,etc
    stock_id = db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=False)
    stock= db.relationship('Stock', backref='news', lazy=True)

class Sentiment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    news_id= db.Column(db.Integer, db.ForeignKey('news.id'), nullable=False)
    news= db.relationship('News', backref='sentiment', lazy=True)
    sentiment= db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, nullable=False)
    sentiment_algorithm= db.Column(db.String(100), nullable=False,default='vader') # vader, textblob, sentiwordnet
    sentiment_score= db.Column(db.Float, nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stock_id= db.Column(db.Integer, db.ForeignKey('stock.id'), nullable=False)
    stock= db.relationship('Stock', backref='prediction', lazy=True)
    predicted_price= db.Column(db.Float, nullable=False)
    predicted_date= db.Column(db.Date, nullable=False)
    prediction_algorithm= db.Column(db.String(100), nullable=False) #linear_regression, arima,LSTM,
    prediction_score= db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)

class ApiKeys(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name= db.Column(db.String(100), nullable=False)
    api_key = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user= db.relationship('User', backref='api_keys', lazy=True)
    date = db.Column(db.Date, nullable=False)


