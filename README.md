# ❤️ Wczesne Wykrywanie Ryzyka Zawału Serca
[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](https://www.repostatus.org/badges/latest/inactive.svg)](https://www.repostatus.org/#inactive)

**Projekt pozwala na wykrywanie ryzyka zawału serca mogącego wystąpić w przeciągu najbliższych 10 lat przy wykorzystaniu modeli uczenia maszynowego. Celem jest opracowanie narzędzi do przewidywania ryzyka na podstawie danych medycznych.**

Repozytorium zawiera:
- interaktywną stronę (Quarto + GitHub Pages) z analizą i prezentacją wyników,
- backend API (FastAPI) do wykonywania predykcji na podstawie danych wejściowych,
- skrypty trenowania i ewaluacji modeli oraz zapisane modele/artefakty.

> [!WARNING]
> Nie jest to narzędzie medyczne i nie powinno być używane do diagnostyki.

## Spis treści
- [Zobacz pełną analizę online](#zobacz-pełną-analizę-online)
- [Informacje ogólne](#informacje-ogólne)
- [Zbiór danych](#zbiór-danych)
- [Użyte technologie](#użyte-technologie)
- [Struktura projektu](#struktura-projektu)
- [Ograniczenia i zastrzeżenia](#ograniczenia-i-zastrzeżenia)
- [Autorzy](#autorzy)

## 🔗 Zobacz pełną analizę online
Analiza projektu wraz z interaktywnymi raportami jest dostępna online:  
**[GitHub Pages – Wczesne Wykrywanie Ryzyka Zawału Serca](https://tgolecki04.github.io/Heart-Disease-Prediction/)**

## ℹ️ Informacje ogólne
Projekt z zakresu analizy danych i modelowania klasyfikacyjnego.  
Celem jest zbudowanie kilku modeli predykcyjnych umożliwiających przewidywanie potencjalnego zawału serca w najbliższych 10 latach na podstawie czynników takich jak m.in.: płeć, wiek, palenie, liczba papierosów dziennie, stosowanie leków, choroby współistniejące (np. nadciśnienie), poziom cholesterolu, ciśnienie, BMI, tętno oraz glukoza.

Modele rozważane/wykorzystane: sieci neuronowe (Keras/TensorFlow/PyTorch), XGBoost, Random Forest oraz warianty z/bez SMOTE.

## 📊 Zbiór danych
Projekt wykorzystuje zbiór danych „Framingham Heart Study” z Kaggle:
**[Framingham Heart Study](https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study)**\
Licencja zbioru danych: **CC0 (Public Domain)**.\
Uwaga: licencja `LICENSE` dotyczy kodu źródłowego projektu. Dane (oraz ewentualne znaki towarowe/nazwy podmiotów trzecich) podlegają własnym warunkom/licencjom.

## 🛠️ Użyte technologie
Zaawansowana analiza danych w języku R. Stworzenie kilku modeli predykcyjnych w Python. Wykorzystanie Quarto do stworzenia spójnego i przejrzystego 
połączenia części teoretycznych i praktycznych projektu.
- Python (trenowanie modeli, predykcja, backend API - FastAPI)
- R (zaawansowana analiza zbioru danych)
- Quarto (raporty, prezentacja, generowanie strony)
- SCSS/CSS/JavaScript (frontend, wizualizacje, interakcje)
- Dodatkowe biblioteki:
  - `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`
  - opcjonalnie: `tensorflow`/`keras`, `torch` (dla niektórych modeli)
- Uwaga: Quarto notebooks (`.qmd`) mogą zawierać kod w Python i R, jednak repozytorium nie zawiera osobnych plików `.R`

## 🗂 Struktura projektu
```
📁 RF_model/                      # Skrypty trenowania/ewaluacji i zapisane modele/artefakty
📁 aplikacja-backend/             # FastAPI backend
📁 models/                        # Bazowe/wytrenowane modele (joblib/pkl) i wykresy
📁 styles/                        # Style SCSS/CSS
📁 images/                        # Grafiki wykorzystywane w raportach/stronie
📁 docs/                          # Wygenerowana strona (GitHub Pages)
📁 data/                          # Dodatkowe dane (jeśli używane)
📄 _quarto.yml                    # Konfiguracja Quarto
📄 index.qmd, wstep.qmd, ...      # Sekcje raportu/strony
📄 aplikacja.qmd                  # Interaktywny formularz z predykcją (frontend)
📄 script.js                      # Skrypt JS strony głównej
📄 README.md
```

## ⚠️ Ograniczenia i zastrzeżenia
- Projekt ma charakter edukacyjny i demonstracyjny. Nie jest certyfikowanym wyrobem medycznym.
- Wyniki predykcji nie powinny być podstawą decyzji diagnostycznych lub terapeutycznych.
- Dane i modele mogą mieć ograniczenia jakościowe wynikające z doboru cech, balansu klas, metod przetwarzania i założeń.

## 👥 Autorzy
- Damian Spodar
- Tomasz Golecki
- Tomasz Hanusek

<a href="https://github.com/tgolecki04/team-project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tgolecki04/team-project"/>
</a>

Kod źródłowy w tym repozytorium jest udostępniany na licencji **Apache License 2.0** — zobacz plik `LICENSE`.
