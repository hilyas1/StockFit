import pandas as pd
import yfinance as yf
from datetime import datetime, date
import h2o
from h2o.automl import H2OAutoML
#import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

#h2o.init()
st.beta_set_page_config(page_title="StockFit",page_icon="📈",layout='wide')

st.title("StockFit")
st.sidebar.subheader("Stock Settings:")
ticker = st.sidebar.text_input('Ticker', value="AAPL")
date_start=st.sidebar.date_input("Start", value=datetime(2016,1,1))
date_end=st.sidebar.date_input("End", value=datetime.today(),max_value=datetime.today())
st.sidebar.subheader("Model Settings:")
tick=yf.Ticker(ticker)
df = tick.history(start=date_start, end=date_end)
try:
    if df.shape[0]>0:
        st.header(tick.info["shortName"])
        st.image(tick.info['logo_url'])
        chart=px.line(df, y="Close")
        chart.update_layout(title=f"{ticker} Stock Close Price Chart", yaxis_title="Closing Price", height=800, width=1200)
        st.plotly_chart(chart)
        st.header("Get Predicted Prices")
        days = st.sidebar.number_input("Days to predict",format="%i", step=1, min_value=1, max_value=30)
        secs = st.sidebar.slider("Model Runtime (seconds)",min_value=30,max_value=120,value=60,format='%i')
        if st.button('Predict') and days >0:
            with st.spinner("Initializing"):
                h2o.init()
            st.success("Inititialized")

            df["Prediction"] = df[['Close']].shift(-days)

            hf = h2o.H2OFrame(df.reset_index())
            #st.dataframe(hf)
            #st.write('Hello')
            length = hf.shape[0]
            split = hf.split_frame(ratios=[.8])
            train = split[0]
            test = split[1]
            y = "Prediction"
            x = hf.columns
            x.remove(y)
            x.remove('Date')
            with st.spinner("Running Machine Learning Model"):
                aml = H2OAutoML(max_runtime_secs=secs, seed=1)
                aml.train(x=x,y=y,training_frame=train.na_omit())
            st.success("Done")
            b=df.reset_index()["Date"]
            d=pd.Series(pd.date_range(start=(pd.to_datetime('today').date()), periods=days))
            d=df.reset_index()["Date"][length:].append(d)
            a=aml.leader.predict(hf[length-days:length,:]).as_data_frame()
            c=hf["Close"].as_data_frame()
            st.spinner("charting")

            #charty = pd.DataFrame({'x':a['predict'].tolist(),'y':d})
            # charto = pd.DataFrame(c,b)
            # st.line_chart(charty)
            # st.line_chart(charto)
            # plt.figure(figsize=(16,10))
            # plt.plot(d,a)
            # plt.plot(b,c)
            # st.pyplot()
            #alt.Chart(data=charty)
            st.table(a.set_index(d))
            fig=px.line(x=b,y=c,)
            fig.add_scatter(x=d,y=a['predict'], name="Predicted Closing Price")
            fig.update_layout(title=f"Predicted Closing prices for the next {days} day(s)", xaxis_title="Date", yaxis_title="Closing price", height=800, width=1200)
            st.plotly_chart(fig)
            h2o.cluster().shutdown()
        else:
            pass
    else:
        st.warning("Ticker not Found")
except Exception as e:
    st.warning(e)


#aml = H2OAutoML(max_runtime_secs=30, seed=1)
#aml.train(x=x,y=y,training_frame=hf.na_omit())

#st.table(df)
#print(aml.leaderboard)


