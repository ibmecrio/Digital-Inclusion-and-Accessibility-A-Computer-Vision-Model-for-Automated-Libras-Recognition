"""
evaluate_model.py — Avaliação canônica do pipeline Libras Vision.

Regenera, de forma determinística, todos os números citados na Seção IV do
artigo a partir do modelo já treinado (models/knn_model.joblib) e do CSV de
teste (landmarks_test.csv):

    - accuracy global, macro/weighted precision/recall/F1
    - relatório por classe (21 letras estáticas do alfabeto LIBRAS)
    - matriz de confusão (CSV + PNG)
    - latência de inferência em CPU
    - baselines de sanidade (DummyClassifier: most_frequent, stratified)

===============================================================================
Como executar
-------------------------------------------------------------------------------
Pré-requisitos:
    - Python 3.11+ (ver requirements.txt)
    - Os artefatos já gerados pelos passos anteriores do projeto:
        landmarks_training.csv          (landmark_extractor.py)
        landmarks_test.csv              (landmark_extractor.py na pasta test/)
        models/knn_model.joblib         (knn_model.py)
        models/label_encoder.joblib     (knn_model.py)

Dependências (mesmas versões do requirements.txt principal):
    scikit-learn, pandas, numpy, joblib        — obrigatórias
    matplotlib                                  — opcional, só para o PNG

Setup num venv isolado (recomendado para não conflitar com o libras_vision.py):

    python3 -m venv .venv
    source .venv/bin/activate                   # macOS/Linux
    # .venv\Scripts\activate                    # Windows PowerShell
    pip install -r requirements.txt

Execução:

    python evaluate_model.py

Saída esperada no terminal:
    - bloco de cabeçalho com contagens de treino/teste e classes
    - accuracy, macro/weighted P/R/F1 e latência média
    - baselines (most_frequent, stratified) — devem ficar em ~1/21 ≈ 4–5 %
    - classification_report completo e top-10 pares confundidos

Artefatos escritos em ./reports/ (criado se não existir):
    summary.json           -> métricas globais + metadados do dataset
    per_class.csv          -> P/R/F1/support por letra
    confusion_matrix.csv   -> matriz 21×21, rows=true, cols=pred
    confusion_matrix.png   -> heatmap pronto para figura do artigo
    confused_pairs.csv     -> erros off-diagonal ordenados por contagem

Reexecução: o script é idempotente — sobrescreve os artefatos a cada run.
Para comparar duas execuções, renomeie ./reports/ antes de rodar de novo.

Troubleshooting rápido:
    InconsistentVersionWarning (joblib)   -> versão de sklearn mudou desde o
                                              treino; regenere os .joblib com
                                              `python knn_model.py`.
    ValueError em load_split()            -> o CSV tem nº de features ≠ 63;
                                              esse CSV foi gerado por outro
                                              extract.py — regenere.
    matplotlib ausente                    -> tudo funciona, só não sai o PNG.

===============================================================================
Biblioteca de avaliação: scikit-learn
-------------------------------------------------------------------------------
Biblioteca de avaliação: scikit-learn
-------------------------------------------------------------------------------
Todas as métricas abaixo vêm do módulo `sklearn.metrics` e do utilitário
`sklearn.dummy.DummyClassifier`. Não é à toa que scikit-learn é a escolha
padrão no meio acadêmico: as fórmulas estão alinhadas com a convenção usada
em Powers (2011) e Sokolova & Lapalme (2009), a documentação oficial cita as
referências, e o zero_division é explícito — o que evita o silêncio
clássico de "precisão 0/0 virou NaN e ninguém notou".

Funções usadas aqui:
    accuracy_score                  — fração de predições corretas.
    precision_recall_fscore_support — P/R/F1/support por classe, com suporte
                                      a average ∈ {None, 'macro', 'weighted',
                                      'micro'}.
    classification_report           — tabela formatada (ou dict se
                                      output_dict=True, útil para exportar).
    confusion_matrix                — matriz NxN de contagens true × predicted;
                                      use labels=... para fixar a ordem.
    DummyClassifier                 — baselines triviais. Aqui servem para
                                      contextualizar "99 % é muito?" — sim,
                                      porque o chute mais frequente fica em
                                      ~5 % (21 classes ~balanceadas).

Referência:
    Pedregosa et al., "Scikit-learn: Machine Learning in Python",
    Journal of Machine Learning Research 12 (2011) 2825–2830.
===============================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# matplotlib é dependência opcional deste script: se o ambiente não tiver
# (por exemplo, um CI minimalista), ainda quero que as métricas saiam. A
# única coisa que perco é o PNG da matriz de confusão.
try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


# Caminhos fixos relativos ao próprio script, não ao CWD — quero que `python
# evaluate_model.py` funcione mesmo se o usuário chamar de outro diretório.
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_CSV    = PROJECT_ROOT / "landmarks_training.csv"
TEST_CSV     = PROJECT_ROOT / "landmarks_test.csv"
MODEL_PATH   = PROJECT_ROOT / "models" / "knn_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
OUT_DIR      = PROJECT_ROOT / "reports"

# Nunca deixo "63" aparecer mágico no meio do código. Se alguém trocar o
# HandLandmarker por um detector que produza outro número de pontos (Pose:
# 33, HolisticLandmarker: 543…), só esta constante muda.
EXPECTED_FEATURES = 21 * 3  # 21 landmarks × (x, y, z)


def load_split(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê um CSV produzido por landmark_extractor.py e devolve (X, y).

    Layout esperado (vem do `pandas.DataFrame.to_csv` sem `index=False`):
        coluna 0           -> índice escrito pelo pandas (descarto)
        colunas 1..63      -> x0,y0,z0, ..., x20,y20,z20
        coluna 64 (última) -> rótulo string ('A', 'B', ...)
    """
    df = pd.read_csv(csv_path)

    # `.to_numpy(dtype=np.float64)` é preferível a `np.array(df.values)`:
    # a primeira respeita os dtypes por coluna sem round-trip por `object`,
    # a segunda já me queimou com colunas mistas virando string silenciosamente.
    X = df.iloc[:, 1:-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy()

    # Se o CSV foi gerado com outra versão do pipeline (ex.: alguém mexeu
    # em extract.py e mudou o número de features), o knn.predict() falha
    # com uma mensagem vaga sobre shape. Melhor explodir aqui, com contexto.
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{csv_path.name}: esperava {EXPECTED_FEATURES} features por "
            f"amostra, obtive {X.shape[1]}. Esse CSV foi gerado pelo mesmo "
            f"extract.py?"
        )
    return X, y


def format_pct(x: float) -> str:
    """Percentual com 2 casas — mesma convenção usada nas tabelas do artigo."""
    return f"{100 * x:.2f}%"


def main() -> None:
    # -------------------------------------------------------------------
    # 1. Dados
    # -------------------------------------------------------------------
    # O treino só é carregado porque os DummyClassifier precisam de um .fit()
    # para definir a distribuição de classes. O KNN em si já foi persistido
    # por knn_model.py, então não reaproveito X_train para nada mais que isso.
    X_train, y_train = load_split(TRAIN_CSV)
    X_test,  y_test  = load_split(TEST_CSV)

    # -------------------------------------------------------------------
    # 2. Modelo treinado + label encoder
    # -------------------------------------------------------------------
    # joblib, não pickle: serializa ndarrays de forma muito mais eficiente
    # (mmap-friendly). Ressalva conhecida: o artefato fica acoplado à versão
    # de scikit-learn do treino — se subir a versão no requirements.txt sem
    # regerar os .joblib, dá o UserWarning de incompatibilidade.
    knn = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)

    # Se o teste contiver alguma letra que não estava no treino, `transform`
    # levanta ValueError. Esse é o comportamento que eu quero: prefiro falhar
    # aqui do que rodar silenciosamente com um mapeamento inconsistente.
    y_test_num = le.transform(y_test)

    # -------------------------------------------------------------------
    # 3. Inferência + latência
    # -------------------------------------------------------------------
    # perf_counter() — relógio monotônico de maior resolução em CPython.
    # Não uso time.time() porque NTP pode fazê-lo andar para trás durante a
    # medição, gerando latência negativa em edge cases.
    #
    # Medição em lote, não em loop: o KNN vetoriza internamente a matriz de
    # distâncias e um loop por amostra penalizaria o modelo injustamente com
    # overhead de Python puro. A latência reportada é elapsed / N.
    t0 = time.perf_counter()
    y_pred = knn.predict(X_test)
    elapsed = time.perf_counter() - t0
    per_sample_ms = (elapsed / len(X_test)) * 1000.0

    # -------------------------------------------------------------------
    # 4. Métricas globais
    # -------------------------------------------------------------------
    acc = accuracy_score(y_test_num, y_pred)

    # zero_division=0: se uma classe nunca for predita, precisão é 0/0.
    # O default dispara um UserWarning — aqui prefiro o valor explícito 0,
    # que já reflete corretamente "nenhuma predição correta daquela classe"
    # e mantém o stdout limpo para ser colado direto no artigo.
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test_num, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test_num, y_pred, average="weighted", zero_division=0
    )

    # -------------------------------------------------------------------
    # 5. Baselines — contextualizam o "99 % é muito?" da Seção IV
    # -------------------------------------------------------------------
    # - most_frequent: sempre chuta a classe dominante. Piso trivial.
    # - stratified:    amostra proporcionalmente à frequência do treino.
    #                  Com 21 classes ~balanceadas, fica em torno de 1/21.
    # random_state fixo para o stratified ser determinístico entre runs.
    baselines: dict[str, float] = {}
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=0)
        dummy.fit(X_train, le.transform(y_train))
        baselines[strategy] = accuracy_score(y_test_num, dummy.predict(X_test))

    # -------------------------------------------------------------------
    # 6. Relatório por classe
    # -------------------------------------------------------------------
    # Preciso dos dois formatos: a string para o stdout (legível), o dict
    # para montar o CSV per_class.csv sem reparsear texto.
    report_dict = classification_report(
        y_test_num, y_pred,
        target_names=list(le.classes_),
        digits=4, zero_division=0, output_dict=True,
    )
    report_str = classification_report(
        y_test_num, y_pred,
        target_names=list(le.classes_),
        digits=4, zero_division=0,
    )

    # No dict, as chaves "accuracy", "macro avg" e "weighted avg" têm schema
    # diferente das classes. Filtro por le.classes_ para empilhar só as linhas
    # que interessam em per_class.csv.
    per_class_rows = []
    for label in le.classes_:
        row = report_dict[label]
        per_class_rows.append({
            "class":     label,
            "precision": row["precision"],
            "recall":    row["recall"],
            "f1":        row["f1-score"],
            "support":   int(row["support"]),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # -------------------------------------------------------------------
    # 7. Matriz de confusão + pares confundidos
    # -------------------------------------------------------------------
    # labels=range(...) força a ordem alfabética (le.classes_ já vem
    # ordenado por fit()). Sem passar labels explicitamente, a ordem de
    # linhas/colunas seria "primeira ocorrência em y_test_num", o que muda
    # entre splits e quebra a comparação do heatmap entre experimentos.
    cm = confusion_matrix(
        y_test_num, y_pred,
        labels=list(range(len(le.classes_))),
    )
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # O que eu realmente quero olhar: só os erros, ordenados por contagem e
    # já com o percentual relativo ao total da classe verdadeira (uma confusão
    # de 18/528 pesa muito diferente de 18/10000 — por isso reporto os dois).
    confused = []
    for i, true_label in enumerate(le.classes_):
        total_true = cm[i].sum()  # suporte efetivo da classe no teste
        for j, pred_label in enumerate(le.classes_):
            if i != j and cm[i, j] > 0:
                confused.append({
                    "true":        true_label,
                    "predicted":   pred_label,
                    "count":       int(cm[i, j]),
                    "pct_of_true": cm[i, j] / total_true,
                })
    confused_df = (
        pd.DataFrame(confused)
          .sort_values("count", ascending=False, ignore_index=True)
    )

    # -------------------------------------------------------------------
    # 8. Persistência dos artefatos
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)

    summary = {
        "dataset": {
            "n_train":    int(len(X_train)),
            "n_test":     int(len(X_test)),
            "n_features": int(X_test.shape[1]),
            "n_classes":  int(len(le.classes_)),
            "classes":    list(le.classes_),
        },
        "knn": {
            "accuracy":              acc,
            "precision_macro":       p_macro,
            "recall_macro":          r_macro,
            "f1_macro":              f1_macro,
            "precision_weighted":    p_w,
            "recall_weighted":       r_w,
            "f1_weighted":           f1_w,
            "latency_ms_per_sample": per_sample_ms,
            "total_inference_s":     elapsed,
        },
        "baselines_accuracy": baselines,
    }

    # default=float cobre np.float64/np.int64 que o json nativo não encoda;
    # indent=2 deixa o summary legível em diff de PR (e no próprio artigo,
    # se for anexado como suplementar).
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    per_class_df.to_csv(OUT_DIR / "per_class.csv", index=False)
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")
    confused_df.to_csv(OUT_DIR / "confused_pairs.csv", index=False)

    # -------------------------------------------------------------------
    # 9. Heatmap da matriz — opcional, só se matplotlib estiver instalado
    # -------------------------------------------------------------------
    # Uso colormap 'Blues' porque se o revisor imprimir em P&B a gradação
    # ainda é distinguível. Anotar só células != 0 e usar texto branco nas
    # células escuras mantém a figura legível mesmo em 150 dpi.
    if _HAS_PLT:
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(le.classes_)))
        ax.set_yticks(range(len(le.classes_)))
        ax.set_xticklabels(le.classes_)
        ax.set_yticklabels(le.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix — test set")
        fig.colorbar(im, ax=ax)

        cm_peak = cm.max()
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                v = cm[i, j]
                if v == 0:
                    continue
                # Threshold de 50 % do pico — empírico, funciona bem no 'Blues'.
                color = "white" if v > cm_peak * 0.5 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        color=color, fontsize=7)

        fig.tight_layout()
        fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    # -------------------------------------------------------------------
    # 10. Resumo no stdout — o que efetivamente vai para a Seção IV
    # -------------------------------------------------------------------
    bar = "=" * 70
    print(bar)
    print("Libras Vision — Avaliação no conjunto de teste")
    print(bar)
    print(f"Treino:  {len(X_train):>6} amostras")
    print(f"Teste:   {len(X_test):>6} amostras  ({X_test.shape[1]} features)")
    print(f"Classes: {len(le.classes_)} -> {', '.join(le.classes_)}")
    print()
    print(f"Accuracy            : {format_pct(acc)}")
    print(f"Macro    P / R / F1 : {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted P / R / F1 : {p_w:.4f} / {r_w:.4f} / {f1_w:.4f}")
    print(f"Latência (KNN, CPU) : {per_sample_ms:.3f} ms/amostra")
    print()
    print("Baselines (accuracy):")
    for name, val in baselines.items():
        print(f"  {name:<15} {format_pct(val)}")
    print()
    print("Relatório por classe:")
    print(report_str)
    print("Top-10 pares confundidos (true -> predicted):")
    for _, row in confused_df.head(10).iterrows():
        print(f"  {row['true']} -> {row['predicted']:<2} "
              f"{int(row['count']):>3}  ({format_pct(row['pct_of_true'])})")
    print()
    print(f"Artefatos salvos em: {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
