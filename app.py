from models import db, User, Stock, StockDetails, Order, Portfolio, News, Sentiment, Prediction, ApiKeys

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
import os
import yfinance as yf
from stock_utils import  analyze_twitter_sentiment, get_news_and_sentiment,analyze_reddit_sentiment
# from predictions import get_stock_predictions
from ds_pre import Predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
predictor = Predictor()

# from models import User, Stock, StockDetails, Order, Portfolio, News, Sentiment, Prediction, ApiKeys

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():

    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('register'))

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        new_user.balance=100000

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    stock_info = None
    try:
        stock_info = request.args.get('stock_info')
    except:
        stock_info = None

    if stock_info:
        """eg. response 
        {'address1': 'One Apple Park Way', 'city': 'Cupertino', 'state': 'CA', 'zip': '95014', 'country': 'United States', 'phone': '408 996 1010', 'website': 'https://www.apple.com', 'industry': 'Consumer Electronics', 'industryKey': 'consumer-electronics', 'industryDisp': 'Consumer Electronics', 'sector': 'Technology', 'sectorKey': 'technology', 'sectorDisp': 'Technology', 'longBusinessSummary': 'Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, a line of smartphones; Mac, a line of personal computers; iPad, a line of multi-purpose tablets; and wearables, home, and accessories comprising AirPods, Apple TV, Apple Watch, Beats products, and HomePod. It also provides AppleCare support and cloud services; and operates various platforms, including the App Store that allow customers to discover and download applications and digital content, such as books, music, video, games, and podcasts. In addition, the company offers various services, such as Apple Arcade, a game subscription service; Apple Fitness+, a personalized fitness service; Apple Music, which offers users a curated listening experience with on-demand radio stations; Apple News+, a subscription news and magazine service; Apple TV+, which offers exclusive original content; Apple Card, a co-branded credit card; and Apple Pay, a cashless payment service, as well as licenses its intellectual property. The company serves consumers, and small and mid-sized businesses; and the education, enterprise, and government markets. It distributes third-party applications for its products through the App Store. The company also sells its products through its retail and online stores, and direct sales force; and third-party cellular network carriers, wholesalers, retailers, and resellers. Apple Inc. was founded in 1976 and is headquartered in Cupertino, California.', 'fullTimeEmployees': 161000, 'companyOfficers': [{'maxAge': 1, 'name': 'Mr. Timothy D. Cook', 'age': 62, 'title': 'CEO & Director', 'yearBorn': 1961, 'fiscalYear': 2023, 'totalPay': 16239562, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Luca  Maestri', 'age': 60, 'title': 'CFO & Senior VP', 'yearBorn': 1963, 'fiscalYear': 2023, 'totalPay': 4612242, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Jeffrey E. Williams', 'age': 59, 'title': 'Chief Operating Officer', 'yearBorn': 1964, 'fiscalYear': 2023, 'totalPay': 4637585, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Ms. Katherine L. Adams', 'age': 59, 'title': 'Senior VP, General Counsel & Secretary', 'yearBorn': 1964, 'fiscalYear': 2023, 'totalPay': 4618064, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': "Ms. Deirdre  O'Brien", 'age': 56, 'title': 'Senior Vice President of Retail', 'yearBorn': 1967, 'fiscalYear': 2023, 'totalPay': 4613369, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Chris  Kondo', 'title': 'Senior Director of Corporate Accounting', 'fiscalYear': 2023, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. James  Wilson', 'title': 'Chief Technology Officer', 'fiscalYear': 2023, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Suhasini  Chandramouli', 'title': 'Director of Investor Relations', 'fiscalYear': 2023, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Greg  Joswiak', 'title': 'Senior Vice President of Worldwide Marketing', 'fiscalYear': 2023, 'exercisedValue': 0, 'unexercisedValue': 0}, {'maxAge': 1, 'name': 'Mr. Adrian  Perica', 'age': 49, 'title': 'Head of Corporate Development', 'yearBorn': 1974, 'fiscalYear': 2023, 'exercisedValue': 0, 'unexercisedValue': 0}], 'auditRisk': 6, 'boardRisk': 1, 'compensationRisk': 2, 'shareHolderRightsRisk': 1, 'overallRisk': 1, 'governanceEpochDate': 1725148800, 'compensationAsOfEpochDate': 1703980800, 'irWebsite': 'http://investor.apple.com/', 'maxAge': 86400, 'priceHint': 2, 'previousClose': 222.38, 'open': 223.9, 'dayLow': 219.77, 'dayHigh': 225.24, 'regularMarketPreviousClose': 222.38, 'regularMarketOpen': 223.9, 'regularMarketDayLow': 219.77, 'regularMarketDayHigh': 225.24, 'dividendRate': 1.0, 'dividendYield': 0.0045, 'exDividendDate': 1723420800, 'payoutRatio': 0.1476, 'fiveYearAvgDividendYield': 0.66, 'beta': 1.24, 'trailingPE': 33.661587, 'forwardPE': 29.52139, 'volume': 47972730, 'regularMarketVolume': 47972730, 'averageVolume': 63432124, 'averageVolume10days': 42719820, 'averageDailyVolume10Day': 42719820, 'bid': 220.78, 'ask': 220.92, 'bidSize': 200, 'askSize': 200, 'marketCap': 3357369434112, 'fiftyTwoWeekLow': 164.08, 'fiftyTwoWeekHigh': 237.23, 'priceToSalesTrailing12Months': 8.706803, 'fiftyDayAverage': 222.575, 'twoHundredDayAverage': 194.7666, 'trailingAnnualDividendRate': 0.97, 'trailingAnnualDividendYield': 0.0043619033, 'currency': 'USD', 'enterpriseValue': 3396880564224, 'profitMargins': 0.26441, 'floatShares': 15179810381, 'sharesOutstanding': 15204100096, 'sharesShort': 121598771, 'sharesShortPriorMonth': 135383184, 'sharesShortPreviousMonthDate': 1721001600, 'dateShortInterest': 1723680000, 'sharesPercentSharesOut': 0.008, 'heldPercentInsiders': 0.02703, 'heldPercentInstitutions': 0.60883, 'shortRatio': 2.19, 'shortPercentOfFloat': 0.008, 'impliedSharesOutstanding': 15309500416, 'bookValue': 4.382, 'priceToBook': 50.392517, 'lastFiscalYearEnd': 1696032000, 'nextFiscalYearEnd': 1727654400, 'mostRecentQuarter': 1719619200, 'earningsQuarterlyGrowth': 0.079, 'netIncomeToCommon': 101956001792, 'trailingEps': 6.56, 'forwardEps': 7.48, 'pegRatio': 2.99, 'lastSplitFactor': '4:1', 'lastSplitDate': 1598832000, 'enterpriseToRevenue': 8.809, 'enterpriseToEbitda': 25.777, '52WeekChange': 0.23115528, 'SandP52WeekChange': 0.20522964, 'lastDividendValue': 0.25, 'lastDividendDate': 1723420800, 'exchange': 'NMS', 'quoteType': 'EQUITY', 'symbol': 'AAPL', 'underlyingSymbol': 'AAPL', 'shortName': 'Apple Inc.', 'longName': 'Apple Inc.', 'firstTradeDateEpochUtc': 345479400, 'timeZoneFullName': 'America/New_York', 'timeZoneShortName': 'EDT', 'uuid': '8b10e4ae-9eeb-3684-921a-9ab27e4d87aa', 'messageBoardId': 'finmb_24937', 'gmtOffSetMilliseconds': -14400000, 'currentPrice': 220.82, 'targetHighPrice': 300.0, 'targetLowPrice': 183.86, 'targetMeanPrice': 240.2, 'targetMedianPrice': 243.0, 'recommendationMean': 2.0, 'recommendationKey': 'buy', 'numberOfAnalystOpinions': 40, 'totalCash': 61801000960, 'totalCashPerShare': 4.065, 'ebitda': 131781001216, 'totalDebt': 101304000512, 'quickRatio': 0.798, 'currentRatio': 0.953, 'totalRevenue': 385603010560, 'debtToEquity': 151.862, 'revenuePerShare': 24.957, 'returnOnAssets': 0.22612, 'returnOnEquity': 1.60583, 'freeCashflow': 86158123008, 'operatingCashflow': 113040998400, 'earningsGrowth': 0.111, 'revenueGrowth': 0.049, 'grossMargins': 0.45962003, 'ebitdaMargins': 0.34175, 'operatingMargins': 0.29556, 'financialCurrency': 'USD', 'trailingPegRatio': 2.1197}
        """
        news_articles=get_news_and_sentiment(symbol, days=7)
        print (news_articles)
        return render_template('dashboard.html', stock_info=stock_info,articles=news_articles)

    # watchlist = current_user.
    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/search_stock', methods=['GET', 'POST'])
@login_required
def search_stock():
    symbol = request.form.get('symbol')
    if not symbol:
        symbol=request.args.get('symbol')

    stock=yf.Ticker(symbol)
    news_articles=get_news_and_sentiment(symbol, days=7)
    tweets=analyze_twitter_sentiment(symbol)
    reddit_posts=analyze_reddit_sentiment(symbol)
    # predictions=get_stock_predictions(symbol)
    short_predictions, short_confidences = predictor.short_term_prediction(symbol,calculate_confidence=True, refresh_models=False)
    long_predictions, long_confidences = predictor.long_term_prediction(symbol,calculate_confidence=True, refresh_models=False)
    predictions={
        'short_predictions': short_predictions,
        'short_confidences': short_confidences,
        'long_predictions': long_predictions,
        'long_confidences': long_confidences
    }
    print(stock.info)
    return render_template('dashboard.html', stock_info=stock.info,articles=news_articles,reddit_posts=reddit_posts,tweets=tweets,predictions=predictions)

@app.route('/add_to_portfolio', methods=['GET', 'POST'])
@login_required
def add_to_portfolio():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol')
        qty=request.form.get('qty')
        qty=float(qty)
        stock_info=yf.Ticker(stock_symbol).info

        if stock_symbol:
            stock = Stock.query.filter_by(symbol=stock_symbol).first()
            if not stock:
                stock = Stock(symbol=stock_symbol, name=stock_info['shortName'])
                db.session.add(stock)
                db.session.commit()
            
            user = User.query.filter_by(id=current_user.id).first()
            #check if stock is already in portfolio
            portfolio = Portfolio.query.filter_by(symbol=stock_symbol, user_id=current_user.id).first()
            if portfolio:
                old_qty = portfolio.quantity
                old_rate=portfolio.price
                new_qty = old_qty + qty
                new_rate = (old_rate*old_qty + qty*stock_info['currentPrice'])/new_qty
                portfolio.quantity = new_qty
                portfolio.price = new_rate
                db.session.add(portfolio)
                db.session.commit()
            else:
                portfolio = Portfolio(symbol=stock_symbol, quantity=qty, price=stock_info['currentPrice'], user_id=current_user.id)
                db.session.add(portfolio)
                db.session.commit()

            user.orders.append(Order(order_type='buy', quantity=qty, price=stock_info['currentPrice'], status='completed', stock=stock))
            
            db.session.commit()
            flash('Stock added to your portfolio!', 'success')

    return redirect(url_for('dashboard'))


@app.route('/remove_from_portfolio', methods=['GET', 'POST'])
@login_required
def remove_from_portfolio():
    if request.method == 'POST':
        stock_symbol = request.form.get('stock_symbol')

        qty=request.form.get('qty')
        qty=float(qty)

        if stock_symbol:
            stock = Stock.query.filter_by(symbol=stock_symbol).first()
            stock_info=yf.Ticker(stock_symbol).info
            if stock:
                portfolio = Portfolio.query.filter_by(symbol=stock_symbol, user_id=current_user.id).first()
                if portfolio:
                    old_qty = portfolio.quantity
                    old_rate=portfolio.price
                    new_qty = old_qty - qty
                    new_rate = (old_rate*old_qty - qty*stock_info['currentPrice'])/new_qty
                    portfolio.quantity = new_qty
                    portfolio.price = new_rate
                    db.session.add(portfolio)
                    db.session.commit()

    return redirect(url_for('dashboard'))
            



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)