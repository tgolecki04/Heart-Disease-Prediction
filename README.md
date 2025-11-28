# ❤️ Wczesne Wykrywanie Ryzyka Zawału Serca
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

**Projekt pozwala na wykrywanie ryzyka zawału serca mogącego wystąpić w przeciągu najbliższych 10 lat przy wykorzystaniu modeli uczenia maszynowego. Celem jest opracowanie narzędzi do przewidywania ryzyka na podstawie danych medycznych.**

## Spis treści
- [Zobacz pełną analizę online](#zobacz-pełną-analizę-online)
- [Informacje ogólne](#informacje-ogólne)
- [Zbiór danych](#zbiór-danych)
- [Technologie](#użyte-technologie)
- [Struktura projektu](#struktura-projektu)
- [Autorzy](#autorzy)

## 🔗 Zobacz pełną analizę online
Analiza projektu wraz z interaktywnymi raportami jest dostępna online:  
**[GitHub Pages – Wczesne Wykrywanie Ryzyka Zawału Serca](https://tgolecki04.github.io/team-project/)**

## ℹ️ Informacje ogólne
Projekt z zakresu analizy danych. Głównym założeniem projektu jest stworzenie nieliniowych modeli predykcyjnych zdolnych do skutecznego 
przewidywania potencjalnego zawału serca w najbliższych 10 latach na podstawie czynników między innymi takich jak płeć, wiek, palenie, 
poziom glukozy, przyjmowane leki oraz poziom cholesterolu. Projekt zakłada stworzenie minimum 2 modeli predykcyjnych, przykładowo pierwszy 
oparty na Neural Networks, a kolejny na Gradient Boosting.

> [!WARNING]
> Projekt jest w fazie aktywnego rozwoju. Wyniki i kod mogą ulegać zmianom, a część funkcjonalności może wymagać dopracowania.

## 📊 Zbiór danych
**[Framingham Heart Study](https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study)**

## 🛠️ Użyte technologie
Zaawansowana analiza danych w języku R. Stworzenie kilku modeli predykcyjnych w Python. Wykorzystanie Quarto do stworzenia spójnego i przejrzystego 
połączenia części teoretycznych i praktycznych projektu.
- R (analiza)
- Python (modele)
- Quarto (raporty i prezentacja)
- SCSS/HTML/JavaScript (frontend, wizualizacje)
- Dodatkowe biblioteki: `tidyverse`, `sklearn`, `ggplot2` itp.

## 🗂 Struktura projektu
```
📄 dane.qmd                # Analiza danych
📄 plan.qmd                # Plan projektu, cele
📄 wnioski.qmd             # Wnioski
📁 _site/                  # Wygenerowane raporty HTML
📁 data/                   # Zbiór danych
📄 README.md
➕ ... (inne pliki .R, .py, .scss, .js itd.)
```

## 👥 Autorzy
- Damian Spodar
- Tomasz Golecki
- Tomasz Hanusek

<a href="https://github.com/tgolecki04/team-project/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tgolecki04/team-project"/>
</a>
