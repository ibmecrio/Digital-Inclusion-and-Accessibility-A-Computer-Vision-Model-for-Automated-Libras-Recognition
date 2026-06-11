"""Render the collected experiment results into a Portuguese technical report."""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def _table(header: List[str], rows: List[List[str]]) -> str:
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def build_report(meta: Dict, comparison: Dict, hyper: Dict,
                 confusion: Dict) -> str:
    L: List[str] = []
    A = L.append

    A("# Relatório Técnico — Avaliação Comparativa dos Classificadores do Libras Vision")
    A("")
    A(f"**Data:** {date.today().isoformat()}  ")
    A(f"**Autor da avaliação:** suíte `evaluation/` (reprodutível via `python -m evaluation.run_all`)  ")
    A(f"**Semente aleatória global:** {meta['random_state']}")
    A("")
    A("---")
    A("")

    # ------------------------------------------------------------------ #
    A("## 1. Objetivo")
    A("")
    A("Avaliar com rigor estatístico os cinco classificadores que o projeto "
      "Libras Vision utiliza para reconhecer o alfabeto manual estático da "
      "LIBRAS a partir de *landmarks* de mão (63 atributos extraídos pelo "
      "MediaPipe). O estudo (i) compara os modelos entre si, (ii) compara "
      "configurações alternativas de hiperparâmetros dentro de cada família e "
      "(iii) caracteriza os erros por meio de matrizes de confusão.")
    A("")

    # ------------------------------------------------------------------ #
    A("## 2. Materiais e métodos")
    A("")
    A("### 2.1 Dados")
    A("")
    A(_table(
        ["Conjunto", "Amostras", "Classes", "Amostras/classe"],
        [["Treino", meta["n_train"], meta["n_classes"], "≈ 200 (194–200)"],
         ["Teste (held-out)", meta["n_test"], meta["n_classes"], "50 (balanceado)"]],
    ))
    A("")
    A(f"O espaço de atributos tem **{meta['n_features']} dimensões** "
      "(21 *landmarks* × 3 coordenadas), as mesmas consumidas em produção por "
      "`libras_vision.py`. As classes são as 20 letras estáticas: "
      f"`{', '.join(meta['classes'])}`. O conjunto de teste é estritamente "
      "*held-out*: nunca é usado para treinar ou selecionar hiperparâmetros.")
    A("")
    A("### 2.2 Protocolo experimental")
    A("")
    A(f"- **Validação cruzada:** `RepeatedStratifiedKFold` "
      f"({comparison['n_splits']} folds × {comparison['n_repeats']} repetições "
      f"= {comparison['n_splits'] * comparison['n_repeats']} estimativas pareadas), "
      "aplicada apenas ao conjunto de treino. Todos os modelos veem exatamente "
      "os mesmos folds, o que torna as pontuações pareadas e habilita testes de "
      "medidas repetidas.")
    A("- **Estimativa de generalização:** cada modelo é re-treinado em todo o "
      "treino e avaliado uma única vez no conjunto de teste *held-out*.")
    A("- **Métricas:** acurácia, e precisão/revocação/F1 macro e ponderado "
      "(macro trata todas as letras igualmente; como o teste é balanceado, "
      "macro e ponderado quase coincidem).")
    A("- **Testes de significância:** Friedman (omnibus sobre os folds) + "
      "Wilcoxon pareado com correção de Holm (post-hoc); McNemar entre os dois "
      "melhores modelos no conjunto de teste.")
    A("- **Reprodutibilidade:** semente global "
      f"`{meta['random_state']}` em todas as divisões e modelos estocásticos.")
    A("")

    # ------------------------------------------------------------------ #
    A("## 3. Comparação entre modelos")
    A("")
    A("### 3.1 Resultados principais")
    A("")
    rows = []
    for label in comparison["ranking"]:
        m = comparison["models"][label]
        rows.append([
            label,
            f"{_pct(m['cv_acc_mean'])} ± {_pct(m['cv_acc_ci95'])}",
            _pct(m["test_accuracy"]),
            f"{m['test_f1_macro']:.4f}",
            f"{m['predict_ms_per_sample']:.3f}",
        ])
    A(_table(
        ["Modelo", "Acurácia CV (média ± IC95%)", "Acurácia teste",
         "F1 macro (teste)", "Predição (ms/amostra)"],
        rows,
    ))
    A("")
    best = comparison["ranking"][0]
    bm = comparison["models"][best]
    A(f"O melhor modelo no conjunto de teste é **{best}** "
      f"(acurácia {_pct(bm['test_accuracy'])}, F1 macro {bm['test_f1_macro']:.4f}).")
    A("")

    A("### 3.2 Significância estatística")
    A("")
    fr = comparison["friedman"]
    A(f"**Teste de Friedman** sobre as pontuações por fold: "
      f"χ² = {fr['statistic']:.2f}, p = {_fmt_p(fr['p_value'])}. "
      + ("Há diferença estatisticamente significativa entre os modelos "
         "(p < 0,05)." if fr["p_value"] < 0.05
         else "Não há evidência de diferença entre os modelos (p ≥ 0,05)."))
    A("")
    A("Ranking médio de Friedman (menor é melhor):")
    A("")
    rank_rows = sorted(fr["avg_ranks"].items(), key=lambda kv: kv[1])
    A(_table(["Modelo", "Rank médio"],
             [[k, f"{v:.2f}"] for k, v in rank_rows]))
    A("")
    A("**Post-hoc Wilcoxon pareado (correção de Holm).** Pares com "
      "p-Holm < 0,05 diferem significativamente:")
    A("")
    ph_rows = []
    for r in sorted(comparison["posthoc_wilcoxon_holm"],
                    key=lambda d: d["p_holm"]):
        ph_rows.append([
            f"{r['a']} vs {r['b']}",
            f"{r['mean_diff'] * 100:+.2f} pp",
            _fmt_p(r["p_holm"]),
            "sim" if r["significant"] else "não",
        ])
    A(_table(["Par", "Δ acurácia CV", "p-Holm", "Significativo?"], ph_rows))
    A("")
    mc = comparison["mcnemar_top2"]
    A(f"**Teste de McNemar** ({mc['model_a']} vs {mc['model_b']}, no teste "
      f"held-out, método {mc['method']}): discordâncias b={mc['b']}, c={mc['c']}; "
      f"p = {_fmt_p(mc['p_value'])}. "
      + ("A diferença entre os dois melhores modelos é significativa."
         if mc["p_value"] < 0.05
         else "Não há diferença significativa entre os dois melhores modelos — "
              "eles são estatisticamente empatados no teste."))
    A("")

    A("### 3.3 Lacuna de generalização (validação cruzada × teste)")
    A("")
    gap_rows = []
    gaps = []
    for label in comparison["ranking"]:
        m = comparison["models"][label]
        gap = (m["cv_acc_mean"] - m["test_accuracy"]) * 100
        gaps.append(gap)
        gap_rows.append([label, _pct(m["cv_acc_mean"]),
                         _pct(m["test_accuracy"]), f"{gap:+.2f} pp"])
    A(_table(["Modelo", "Acurácia CV", "Acurácia teste", "Lacuna"], gap_rows))
    A("")
    A(f"Todos os modelos exibem uma queda consistente de ~{np.mean(gaps):.1f} pp "
      "da validação cruzada (≈ 99–100%) para o teste *held-out* (≈ 90–94%). "
      "Uma lacuna desse tamanho, sistemática em todas as famílias, é o achado "
      "mais importante do estudo: ela indica que a validação cruzada sobre o "
      "treino é **otimista**. A causa mais provável é a presença de quadros "
      "altamente correlacionados no treino (imagens consecutivas do mesmo gesto/"
      "pessoa), que fazem com que cada fold de CV contenha quase-duplicatas das "
      "amostras de validação — inflando a acurácia. O teste *held-out*, "
      "capturado de forma independente, é portanto a estimativa de "
      "generalização confiável; a CV deve ser lida apenas como sinal **relativo** "
      "de ordenação entre modelos/configurações, não como acurácia esperada em "
      "produção.")
    A("")

    # ------------------------------------------------------------------ #
    A("## 4. Comparação de configurações (hiperparâmetros)")
    A("")
    A("Cada família foi varrida por validação cruzada estratificada "
      f"({list(hyper.values())[0]['n_splits']} folds) no conjunto de treino. "
      "A configuração de produção (repo) está destacada. O teste *held-out* "
      "**não** foi usado nesta etapa, para evitar vazamento.")
    A("")
    for family, res in hyper.items():
        A(f"### 4.{list(hyper).index(family) + 1} {family}")
        A("")
        rows = []
        # Show top 5 plus the repo default if it isn't already there.
        shown = res["rows"][:5]
        if res["repo_default"] and res["repo_default"] not in shown:
            shown = shown + [res["repo_default"]]
        for r in shown:
            tag = " ⟵ repo" if r["is_repo_default"] else ""
            rows.append([
                r["rank"],
                r["config"] + tag,
                f"{_pct(r['acc_mean'])} ± {_pct(r['acc_ci95'])}",
            ])
        A(_table(["#", "Configuração", "Acurácia CV (média ± IC95%)"], rows))
        A("")
        best = res["best"]
        repo = res["repo_default"]
        if repo:
            delta = (best["acc_mean"] - repo["acc_mean"]) * 100
            if best["is_repo_default"]:
                A(f"A configuração de produção **já é a melhor** da varredura "
                  f"({_pct(repo['acc_mean'])}).")
            else:
                A(f"Melhor configuração: `{best['config']}` "
                  f"({_pct(best['acc_mean'])}), "
                  f"{delta:+.2f} pp em relação ao repo "
                  f"(`{repo['config']}`, {_pct(repo['acc_mean'])}). "
                  + ("Diferença dentro do intervalo de confiança — sem ganho "
                     "estatístico claro." if delta < repo["acc_ci95"] * 100
                     else "Ganho potencial relevante."))
        A("")

    # ------------------------------------------------------------------ #
    A("## 5. Análise de erros — matrizes de confusão")
    A("")
    A("As matrizes de confusão normalizadas por linha de cada modelo estão em "
      "`evaluation/figures/`. Abaixo, os pares mais confundidos e as letras "
      "mais difíceis por modelo no conjunto de teste.")
    A("")
    for label in comparison["ranking"]:
        if label not in confusion:
            continue
        c = confusion[label]
        A(f"### {label}")
        A("")
        A(f"![Matriz de confusão {label}]({c['figure'].replace(chr(92), '/')})")
        A("")
        if c["top_confusions"]:
            conf_str = "; ".join(
                f"{d['true']}→{d['pred']} ({d['count']}x, {_pct(d['rate'])})"
                for d in c["top_confusions"][:5])
            A(f"**Principais confusões:** {conf_str}.")
        else:
            A("**Principais confusões:** nenhuma — classificação perfeita no teste.")
        A("")
        worst_str = ", ".join(
            f"{d['class']} (F1={d['f1']:.2f})" for d in c["worst_classes"])
        A(f"**Letras com menor F1:** {worst_str}.")
        A("")

    # ------------------------------------------------------------------ #
    A("## 6. Discussão e recomendações")
    A("")
    A(f"1. **Modelo recomendado para produção:** {best_recommendation(comparison)}.")
    A("2. **Configurações:** ver Seção 4 — quando a melhor configuração está "
      "dentro do IC95% da de produção, manter o repo é defensável por "
      "simplicidade/velocidade. Destaque: o KNN de produção (`k=21`) fica "
      "~2 pp abaixo de valores menores de `k`; reduzir `k` melhoraria a "
      "acurácia, ao custo do índice de confiança suave que motivou `k=21`.")

    # Mean per-class F1 across all models -> the systematically hardest letters.
    class_f1: Dict[str, List[float]] = {}
    for model_res in confusion.values():
        for pc in model_res["per_class"]:
            class_f1.setdefault(pc["class"], []).append(pc["f1"])
    mean_f1 = {c: float(np.mean(v)) for c, v in class_f1.items()}
    worst = sorted(mean_f1.items(), key=lambda kv: kv[1])[:4]
    worst_str = ", ".join(f"**{c}** (F1 médio {v:.2f})" for c, v in worst)
    A(f"3. **Erro sistemático crítico:** a letra {worst[0][0]} é mal "
      "classificada por **todos** os modelos. As letras mais difíceis, em "
      f"média entre os cinco classificadores, são: {worst_str}. O padrão "
      "dominante é a confusão R→U / R→V (letras de configuração de mão muito "
      "próximas — dedos cruzados/justapostos), além de I→A. Nenhum classificador "
      "resolve isso, o que indica que o problema está nos **atributos**, não no "
      "modelo: os 63 *landmarks* normalizados não separam bem essas formas. "
      "Recomenda-se aumento de dados para R/U/V e/ou atributos adicionais "
      "(ângulos entre dedos, distâncias inter-falange) antes de trocar de modelo.")
    A("4. **Limitações:** a avaliação usa *landmarks* já extraídos (não mede o "
      "erro do detector MediaPipe nem condições de iluminação/câmera reais), e "
      "as letras dinâmicas (H, J, K, X, Z) estão fora de escopo. A acurácia de "
      "validação cruzada é otimista (Seção 3.3); use o teste *held-out* como "
      "referência de generalização.")
    A("")
    A("## 7. Reprodutibilidade")
    A("")
    A("```bash\npython -m evaluation.run_all\n```")
    A("")
    A("Gera este relatório (`RELATORIO_TECNICO.md`), as figuras em "
      "`evaluation/figures/` e os resultados brutos em "
      "`evaluation/results.json`.")
    A("")

    return "\n".join(L)


def best_recommendation(comparison: Dict) -> str:
    best = comparison["ranking"][0]
    bm = comparison["models"][best]
    mc = comparison["mcnemar_top2"]
    if mc["p_value"] >= 0.05:
        second = comparison["ranking"][1]
        return (f"**{best}** lidera ({_pct(bm['test_accuracy'])} no teste), mas "
                f"empata estatisticamente com **{second}** (McNemar p="
                f"{_fmt_p(mc['p_value'])}); a escolha pode considerar custo de "
                f"predição ({bm['predict_ms_per_sample']:.3f} ms/amostra)")
    return (f"**{best}** ({_pct(bm['test_accuracy'])} no teste), superior aos "
            "demais com significância estatística")
