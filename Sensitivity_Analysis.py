import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def ttest_param_vs_target(
    param_array,
    target_array,
    feature_names,
    target_label="Target",
    ax=None,
    thr=2.0,
    bar_color="skyblue"
):
    """
    param_array : (n_samples, n_features) 스케일된 파라미터 배열
    target_array: (n_samples,) 타깃 값 (BHP mean, TVERDPLGEO mean 등)
    feature_names: 각 열 이름(list-like)
    target_label: 그래프 제목에 들어갈 이름
    ax: 미리 만든 axes 넣을 때
    thr: t-통계량 유의성 기준선 (보통 2)
    bar_color: 막대 색
    """
    param_array = np.asarray(param_array)
    target_array = np.asarray(target_array)

    # high / low 그룹 나누기
    mask_high = target_array > target_array.mean()
    mask_low  = ~mask_high

    t_stats = []
    for i in range(param_array.shape[1]):
        high_vals = param_array[mask_high, i]
        low_vals  = param_array[mask_low, i]
        t_stat, _ = ttest_ind(high_vals, low_vals, equal_var=False)
        t_stats.append(t_stat)

    t_series = pd.Series(t_stats, index=feature_names)
    # 절댓값 기준 정렬
    t_series = t_series.reindex(t_series.abs().sort_values(ascending=False).index)

    # 그리기
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(t_series.index, t_series.values,
                  color=bar_color, edgecolor="black")

    # 값 표시
    for bar, value in zip(bars, t_series.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value > 0 else "top",
                fontsize=8)

    # 기준선
    ax.axhline(y= thr, color="red", linestyle="--", linewidth=1)
    ax.axhline(y=-thr, color="red", linestyle="--", linewidth=1)

    ax.set_title(f"t-test: {target_label}")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("t-statistic")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    return t_series

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# ① Feature Importance 계산 함수
# ──────────────────────────────────────────────
def rf_importance_param_vs_target(param_array, target_array, feature_names, random_state=42):
    """
    Random Forest 기반 feature importance 계산

    param_array : (n_samples, n_features)
    target_array: (n_samples,)
    feature_names : list-like
    """
    model = RandomForestRegressor(random_state=random_state)
    model.fit(param_array, target_array)
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)
    return importance


# ──────────────────────────────────────────────
# ② Feature Importance 시각화 함수
# ──────────────────────────────────────────────
def plot_rf_importance(importance_series, target_label="Target", ax=None, color="skyblue"):
    """
    Random Forest Feature Importance 시각화

    importance_series : pd.Series (index = feature names)
    target_label : subplot 제목
    ax : matplotlib axis (없으면 새로 생성)
    color : 막대 색상
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x=importance_series.values,
        y=importance_series.index,
        ax=ax,
        color=color,
        edgecolor="black"
    )

    ax.set_title(f"Random Forest: {target_label}")
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Parameter")
    ax.grid(True, linestyle="--", linewidth=0.5)

    return ax

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def lasso_paths_param_vs_target(
    param_array,
    target_array,
    feature_names,
    lambdas=None,
    max_iter=10000
):
    """
    여러 alpha(=lambda)에 대해 Lasso를 돌려서 coefficient path를 구하는 함수

    param_array : (n_samples, n_features)
    target_array: (n_samples,)
    feature_names: list of str
    lambdas : iterable of alphas (default: logspace(-7, 2, 30))
    return: lambdas, coefs (shape = [n_lambdas, n_features]), feature_names
    """
    if lambdas is None:
        lambdas = np.logspace(-7, 2, 30)

    coefs = []
    for alpha in lambdas:
        model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(param_array, target_array)
        coefs.append(model.coef_)

    coefs = np.array(coefs)  # (n_lambdas, n_features)
    return np.array(lambdas), coefs, list(feature_names)


def plot_lasso_paths(lambdas, coefs, feature_names, ax=None,
                     title="Lasso paths", show_legend=False, legend_ncol=1):
    """
    Lasso coefficient path를 그리는 함수

    lambdas : (n_lambdas,)
    coefs   : (n_lambdas, n_features)
    feature_names : list[str]
    ax : matplotlib axis
    show_legend : bool
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for i, name in enumerate(feature_names):
        ax.plot(lambdas, coefs[:, i], label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)

    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, ncol=legend_ncol)

    return ax

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_lasso_extinction_index(coefs, lambdas, threshold=1e-6):
    """
    각 파라미터별로 coefficient가 0에 도달하는 첫 번째 lambda index 계산

    coefs : (n_lambdas, n_features)
    lambdas : (n_lambdas,)
    threshold : float, 0으로 간주할 기준 (default 1e-6)

    return : pd.Series (index=feature_names, values=zero_index)
    """
    zero_idx = []
    for i in range(coefs.shape[1]):
        coef_path = coefs[:, i]
        zero_lambda_index = np.argmax(np.abs(coef_path) < threshold)
        # 완전히 0이 되는 지점이 없으면 마지막으로 처리
        if np.abs(coef_path[zero_lambda_index]) >= threshold:
            zero_lambda_index = len(lambdas)
        zero_idx.append(zero_lambda_index)
    return np.array(zero_idx)


def plot_lasso_extinction(zero_idx, feature_names, target_label="Target",
                          ax=None, color="skyblue"):
    """
    Lasso Extinction Speed 시각화

    zero_idx : (n_features,) 각 feature의 zero index
    feature_names : list[str]
    target_label : subplot 제목
    """
    # 정렬
    sorted_idx = np.argsort(zero_idx)
    features_sorted = np.array(feature_names)[sorted_idx]
    extinction_scores = zero_idx[sorted_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x=features_sorted, y=extinction_scores,
        color=color, edgecolor="black", ax=ax
    )
    ax.set_title(f"Lasso Extinction Speed: {target_label}", fontsize=13)
    ax.set_ylabel("Lambda Index of First Zero Coefficient")
    ax.set_xlabel("Parameters")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax

import pandas as pd
import numpy as np

def corr_param_vs_targets(param_df, target_bhp, target_insar,
                          name_bhp="BHP_Mean", name_insar="TVERDPLGEO_mean"):
    """
    파라미터 DataFrame과 두 개의 타깃을 받아서
    각 타깃과의 상관계수 Series 두 개를 반환.

    param_df : 파라미터만 있는 DataFrame (index = samples, cols = features)
    target_bhp : (n_samples,)
    target_insar : (n_samples,)
    """
    df_tmp = param_df.copy()
    df_tmp[name_insar] = target_insar
    df_tmp[name_bhp] = target_bhp

    corr = df_tmp.corr()

    corr_insar = corr[name_insar].drop([name_insar, name_bhp])
    corr_bhp   = corr[name_bhp].drop([name_insar, name_bhp])

    return corr_bhp, corr_insar



import matplotlib.pyplot as plt

def plot_corr_two_targets(corr_bhp, corr_insar,
                          label_bhp="BHP",
                          label_insar="Surface deformation",
                          figsize=(12, 4)):
    """
    두 개의 상관계수 Series를 좌우로 그려주는 함수
    corr_bhp, corr_insar : pd.Series (index = feature names)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # BHP
    corr_bhp.sort_values().plot(
        kind="barh",
        ax=axes[0],
        color="lightcoral",
        edgecolor="black"
    )
    axes[0].set_title(f"Correlation with {label_bhp}")
    axes[0].set_xlabel("Correlation Coefficient")
    axes[0].grid(True, linestyle="--", linewidth=0.5)

    # InSAR / Surface deformation
    corr_insar.sort_values().plot(
        kind="barh",
        ax=axes[1],
        color="skyblue",
        edgecolor="black"
    )
    axes[1].set_title(f"Correlation with {label_insar}")
    axes[1].set_xlabel("Correlation Coefficient")
    axes[1].grid(True, linestyle="--", linewidth=0.5)

    plt.suptitle("Correlation of BHP vs TVERDPLGEO", fontsize=16)
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def build_sensitivity_table(
    feature_names,
    t_bhp,
    t_insar,
    imp_bhp,
    imp_insar,
    ext_bhp,
    ext_insar,
    corr_bhp,
    corr_insar,
):
    """
    여러 민감도 지표(t-test, RF, Lasso-extinction, Correlation)를
    하나의 DataFrame으로 통합하고 0~1 정규화까지 해서 리턴.

    feature_names : list[str]
    t_bhp, t_insar : pd.Series or array-like
    imp_bhp, imp_insar : pd.Series or array-like
    ext_bhp, ext_insar : pd.Series or array-like
    corr_bhp, corr_insar : pd.Series or array-like (보통 abs 값)
    """
    # pd.Series로 통일
    t_bhp = pd.Series(t_bhp, index=feature_names)
    t_insar = pd.Series(t_insar, index=feature_names)
    imp_bhp = pd.Series(imp_bhp, index=feature_names)
    imp_insar = pd.Series(imp_insar, index=feature_names)
    ext_bhp = pd.Series(ext_bhp, index=feature_names)
    ext_insar = pd.Series(ext_insar, index=feature_names)
    corr_bhp = pd.Series(corr_bhp, index=feature_names)
    corr_insar = pd.Series(corr_insar, index=feature_names)

    # extinction은 값이 작을수록 "빨리 죽는" 거라서
    # 그대로 쓸 건지, 역수를 쓸 건지 상황에 따라 바꿀 수 있음.
    # 지금은 네 코드 그대로 "작은 값 = 중요" 로 두고 MinMax에 맡길게.
    df_all = pd.concat(
        [
            t_bhp.abs().rename("t_bhp"),
            t_insar.abs().rename("t_insar"),
            imp_bhp.rename("imp_bhp"),
            imp_insar.rename("imp_insar"),
            ext_bhp.rename("ext_bhp"),
            ext_insar.rename("ext_insar"),
            corr_bhp.abs().rename("corr_bhp"),
            corr_insar.abs().rename("corr_insar"),
        ],
        axis=1,
    )

    # 0~1 스케일
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_all),
        index=feature_names,
        columns=df_all.columns,
    )

    # 타깃별 평균 점수
    df_scaled["score_bhp"] = df_scaled[["t_bhp", "imp_bhp", "ext_bhp", "corr_bhp"]].mean(axis=1)
    df_scaled["score_insar"] = df_scaled[["t_insar", "imp_insar", "ext_insar", "corr_insar"]].mean(axis=1)

    # 전체 평균
    df_scaled["sensitivity_score"] = df_scaled[
        [
            "t_bhp", "t_insar",
            "imp_bhp", "imp_insar",
            "ext_bhp", "ext_insar",
            "corr_bhp", "corr_insar",
        ]
    ].mean(axis=1)

    return df_scaled

def plot_sensitivity_summary(df_scaled, title="Sensitivity Scores by Target Variable and Overall"):
    """
    df_scaled : build_sensitivity_table()에서 나온 DataFrame
                index = feature_names
                columns 에 score_bhp, score_insar, sensitivity_score 있어야 함
    """
    score_bhp = df_scaled["score_bhp"].sort_values(ascending=False)
    score_insar = df_scaled["score_insar"].sort_values(ascending=False)
    overall = df_scaled["sensitivity_score"].sort_values(ascending=False)

    colors = cm.viridis(np.linspace(0, 1, len(overall)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) BHP
    score_bhp.plot(kind="barh", ax=axes[0], color="lightcoral", edgecolor="black")
    axes[0].set_title("BHP Sensitivity Score", fontsize=14)
    axes[0].set_xlabel("Normalized Score")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle='--', linewidth=0.5)

    # (b) InSAR / Surface deformation
    score_insar.plot(kind="barh", ax=axes[1], color="skyblue", edgecolor="black")
    axes[1].set_title("Surface Deformation Sensitivity Score", fontsize=14)
    axes[1].set_xlabel("Normalized Score")
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle='--', linewidth=0.5)

    # (c) Overall
    overall.sort_values().plot(
        kind="barh",
        ax=axes[2],
        color=colors,
        edgecolor="black",
        alpha=0.5
    )
    axes[2].set_title("Overall Sensitivity Score by Feature", fontsize=14)
    axes[2].set_xlabel("Normalized Score")
    axes[2].grid(True, linestyle='--', linewidth=0.5)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
