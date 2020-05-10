import pandas as pd
from fbprophet import Prophet
from datetime import datetime


def fetch_data():
    data = pd.read_csv('/data/aerie-funnel.csv')
    return data


def prepare_data():
    data = fetch_data()
    data['event_date'] = pd.to_datetime(data['event_date'], format="%Y-%m-%d")

    data['unique_visitors'] = data['unique_visitors'].replace(
        ',', '', regex=True).astype(int)
    data['product_visitors'] = data['product_visitors'].replace(
        ',', '', regex=True).astype(int)
    data['segment_visitors'] = data['segment_visitors'].replace(
        ',', '', regex=True).astype(int)
    data['rfqs'] = data['rfqs'].replace(',', '', regex=True).astype(int)
    data['sales'] = data['sales'].replace(',', '', regex=True).astype(int)

    # Aggregate over day
    sorteddata = data.sort_values('event_date')
    funnel = sorteddata.set_index('event_date')  # .drop('mkt_medium')
    agg = funnel.groupby('event_date').sum()
    return agg.reset_index()


def ratio_calculator(df, numerator, denominator, agg_date, ratio_name):
    ratio = df[[agg_date, numerator, denominator]]
    ratio[ratio_name] = ratio.apply(
        lambda row: row[numerator] / row[denominator], axis=1)
    return ratio


def driver_calculator(df, left, right, metric_name):
    df[metric_name] = df.apply(lambda row: row[left] * row[right], axis=1)
    return df


def get_forecast(df, metric, date_field, period_in_days, seasonality=True):
    m = Prophet(yearly_seasonality=seasonality)
    data = df.rename(columns={date_field: 'ds', metric: 'y'})[
        ['ds', 'y']]
    m.fit(data)

    future = m.make_future_dataframe(periods=period_in_days)
    forecast = m.predict(future)
    result = forecast[['ds', 'yhat']].rename(columns={'yhat': metric})
    return result


def get_baseline(seasonality):
    df = prepare_data()

    # Forecast data for UV
    forecast_uv = get_forecast(
        df, 'unique_visitors', 'event_date', seasonality, 365)

    # Calculate UV PV ratio
    uv_pv_ratio = ratio_calculator(
        df, 'product_visitors', 'unique_visitors', 'event_date', 'uv_pv')

    # Forecast UV PV Ratio
    forecast_uv_pv = get_forecast(uv_pv_ratio, 'uv_pv', 'event_date', seasonality, 365)

    # Merge Drivers back together
    df_merged = pd.merge(forecast_uv, forecast_uv_pv, how='inner', on='ds')

    # Calculate output of drivers
    result = driver_calculator(
        df_merged, 'unique_visitors', 'uv_pv', 'product_visitors')
    import pickle
    pickle.dump(result, open('baseline.pkl', 'wb'))
    return 0


# def apply_adjustment(baseline, start_date, drivers, year_key, month_key): #TODO refactor
#     # baseline = baseline.copy()
#     for name, adjustment in drivers.items():
#         def date_adjustment(row):
#             if row.ds.month==month_key and row.ds.year==year_key:
#                 return row[name] * (1+(adjustment/100))
#             else:
#                 return row[name]

#         baseline[name] = baseline.apply(date_adjustment, axis=1)

#     # Apply drivers to one another
#     baseline['pv'] = baseline.apply(lambda row: row.uv*row.uv_pv_ratio, axis=1)
#     baseline = baseline[baseline.ds>=start_date]

#     metrics = ["uv","pv"]
#     for m in metrics:
#         baseline[m] = baseline[m].map(int)
#         ax_v = baseline[['ds',m]].set_index('ds').plot(figsize=(9,5))
#         fig_v = ax_v.get_figure()
#         fig_v.savefig(f'/app/budgetApp/static/{m}_data.png')

#     ax = baseline[['ds','uv','pv']].set_index('ds').plot(figsize=(9,5))
#     fig = ax.get_figure()
#     fig.savefig('/app/budgetApp/static/budget_data.png')

#     return baseline[['ds','uv','pv']]
if __name__ == '__main__':
    get_baseline(True)
