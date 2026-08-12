import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['OrderID', 'CustomerID', 'TrackingNumber', 'Date'])
    df['CouponCode'] = df['CouponCode'].fillna('NoCoupon')
    return df

def encode_features(df):
    categorical_cols = ['Product', 'ShippingAddress', 'PaymentMethod', 'CouponCode', 'ReferralSource']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    le = LabelEncoder()
    df_encoded['OrderStatus'] = le.fit_transform(df_encoded['OrderStatus'])
    return df_encoded, le