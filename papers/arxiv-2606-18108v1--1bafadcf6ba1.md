---
identifier: arxiv:2606.18108v1
title: "Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"
authors:
  - P. A. Estevez
  - J. Espejo-Moreira
  - S. Sanfeliu-Alvarez
  - F. Forster
  - A. M. Munoz Arancibia
  - G. Cabrera-Vives
  - F. E. Bauer
  - A. Bayo
  - M. Catelan
  - R. Dastidar
  - L. Hernandez-Garcia
  - J. A. Intriago
  - G. Pignata
published: "2026-06-16T16:12:16+00:00"
url: https://arxiv.org/abs/2606.18108v1
source: arxiv
doi: null
arxiv_id: 2606.18108v1
categories:
  - astro-ph.IM
  - cs.AI
---

# Querying an astronomical database using large language models: the ALeRCE text-to-SQL system

P.A. Estévez
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0001-9164-4722 "ORCID 0000-0001-9164-4722")
Affiliation: Department of Electrical Engineering, University of Chile,
Av. Tupper 2007, Santiago, Chile Affiliation: Millennium Institute of
Astrophysics (MAS), Nuncio Monseñor Sótero Sanz 100, Providencia,
Santiago, Chile    J. Espejo-Moreira
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0009-0004-2094-7303 "ORCID 0009-0004-2094-7303")
Affiliation: Department of Electrical Engineering, University of Chile,
Av. Tupper 2007, Santiago, Chile Affiliation: Millennium Institute of
Astrophysics (MAS), Nuncio Monseñor Sótero Sanz 100, Providencia,
Santiago, Chile    S. Sanfeliú-Alvarez Affiliation: Department of
Electrical Engineering, University of Chile, Av. Tupper 2007, Santiago,
Chile Affiliation: Departamento de Astronomía, Universidad de Chile,
Casilla 36D, Santiago, Chile    F. Förster
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0003-3459-2270 "ORCID 0000-0003-3459-2270")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile Affiliation: Data and
Artificial Intelligence Initiative (ID&IA), Universidad de Chile
Affiliation: Center for Mathematical Modeling, Universidad de Chile,
Beauchef 851, North building, 7th floor, Santiago 8320000, Chile
Affiliation: Departamento de Astronomía, Universidad de Chile, Casilla
36D, Santiago, Chile    A. M. Muñoz Arancibia
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0002-8722-516X "ORCID 0000-0002-8722-516X")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile Affiliation: Center for
Mathematical Modeling, Universidad de Chile, Beauchef 851, North
building, 7th floor, Santiago 8320000, Chile    G. Cabrera-Vives
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0002-2720-7218 "ORCID 0000-0002-2720-7218")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile Affiliation: Department of
Computer Science, Universidad de Concepción, Edmundo Larenas 219,
Concepción, Chile Affiliation: Center for Data and Artificial
Intelligence, Universidad de Concepción, Edmundo Larenas 310,
Concepción, Chile Affiliation: Heidelberg Institute for Theoretical
Studies, Heidelberg, Baden-Württemberg, Germany    F. E. Bauer
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0002-8686-8737 "ORCID 0000-0002-8686-8737")
Affiliation: Instituto de Alta Investigación, Universidad de Tarapacá,
Casilla 7D, Arica, 1010000, Chile    A. Bayo
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0001-7868-7031 "ORCID 0000-0001-7868-7031")
Affiliation: European Southern Observatory, Karl-Schwarzschild-Strasse
2, 85748 Garching bei München, Germany    M. Catelan
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0001-6003-8877 "ORCID 0000-0001-6003-8877")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile Affiliation: Instituto de
Astrofísica, Facultad de Física, Pontificia Universidad Católica de
Chile, Casilla 306, Santiago 22, Chile Affiliation: Centro de
Astroingeniería, Pontificia Universidad Católica de Chile, Av. Vicuña
Mackenna 4860, 7820436 Macul, Santiago, Chile    R. Dastidar
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0001-6191-7160 "ORCID 0000-0001-6191-7160")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile    L. Hernández-García
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0002-8606-6961 "ORCID 0000-0002-8606-6961")
Affiliation: Millennium Institute of Astrophysics (MAS), Nuncio Monseñor
Sótero Sanz 100, Providencia, Santiago, Chile Affiliation: Instituto de
Estudios Astrofísicos, Facultad de Ingeniería y Ciencias, Universidad
Diego Portales, Av. Ejército Libertador 441, Santiago, Chile
Affiliation: Centro Interdisciplinario de Data Science, Facultad de
Ingeniería y Ciencias, Universidad Diego Portales, Av. Ejército
Libertador 441, Santiago, Chile    J.A. Intriago
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0001-8964-7385 "ORCID 0000-0001-8964-7385")
Affiliation: Department of Electrical Engineering, University of Chile,
Av. Tupper 2007, Santiago, Chile    G. Pignata
[![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X29yY2lkbG9nbyIgaGVpZ2h0PSIxZW0iIHZlcnNpb249IjEuMSIgdmlld2JveD0iMCAwIDcyIDcyIiB3aWR0aD0iMWVtIj48cGF0aCBzdHlsZT0iLS1sdHgtZmlsbC1jb2xvcjojQTZDRTM5OyIgZD0iTTcyLDM2IEM3Miw1NS44ODQzNzUgNTUuODg0Mzc1LDcyIDM2LDcyIEMxNi4xMTU2MjUsNzIgMCw1NS44ODQzNzUgMCwzNiBDMCwxNi4xMTU2MjUgMTYuMTE1NjI1LDAgMzYsMCBDNTUuODg0Mzc1LDAgNzIsMTYuMTE1NjI1IDcyLDM2IFoiIGZpbGw9IiNBNkNFMzkiIC8+PGcgc3R5bGU9Ii0tbHR4LWZpbGwtY29sb3I6I0ZGRkZGRjsiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4Ljg2ODk2NiwgMTIuOTEwMzQ1KSI+PHBvbHlnb24gcG9pbnRzPSI1LjAzNzM0OTI5IDM5LjEyNTA4NzggMC42OTU0Mjk4NjEgMzkuMTI1MDg3OCAwLjY5NTQyOTg2MSA5LjE0NDMxNzg3IDUuMDM3MzQ5MjkgOS4xNDQzMTc4NyA1LjAzNzM0OTI5IDIyLjY5MzA1MDUgNS4wMzczNDkyOSAzOS4xMjUwODc4Ij48L3BvbHlnb24+PHBhdGggZD0iTTExLjQwOTI1Nyw5LjE0NDMxNzg3IEwyMy4xMzgwNzg0LDkuMTQ0MzE3ODcgQzM0LjMwMzAxNCw5LjE0NDMxNzg3IDM5LjIwODgxOTEsMTcuMDY2NDA3NCAzOS4yMDg4MTkxLDI0LjE0ODY5OTUgQzM5LjIwODgxOTEsMzEuODQ2ODQzIDMzLjE0NzA0ODUsMzkuMTUzMDgxMSAyMy4xOTQ0NjY5LDM5LjE1MzA4MTEgTDExLjQwOTI1NywzOS4xNTMwODExIEwxMS40MDkyNTcsOS4xNDQzMTc4NyBaIE0xNS43NTExNzY1LDM1LjI2MjAxOTQgTDIyLjY1ODc3NTYsMzUuMjYyMDE5NCBDMzIuNDk4NTgsMzUuMjYyMDE5NCAzNC43NTQxMjI2LDI3Ljg0MzgwODQgMzQuNzU0MTIyNiwyNC4xNDg2OTk1IEMzNC43NTQxMjI2LDE4LjEzMDE1MDkgMzAuODkxNTA1OSwxMy4wMzUzNzk1IDIyLjQzMzIyMTMsMTMuMDM1Mzc5NSBMMTUuNzUxMTc2NSwxMy4wMzUzNzk1IEwxNS43NTExNzY1LDM1LjI2MjAxOTQgWiIgLz48cGF0aCBkPSJNNS43MTQwMTIwNiwyLjkwMTgyMzI5IEM1LjcxNDAxMjA2LDQuNDQxNDUyIDQuNDQ1MjY5MzcsNS43MjkxNDE0NiAyLjg2NjM4OTU4LDUuNzI5MTQxNDYgQzEuMjg3NTA5NzgsNS43MjkxNDE0NiAwLjAxODc2NzA5MTgsNC40NDE0NTIgMC4wMTg3NjcwOTE4LDIuOTAxODIzMjkgQzAuMDE4NzY3MDkxOCwxLjMzNDIwMTMzIDEuMjg3NTA5NzgsMC4wNzQ1MDUxMDk2IDIuODY2Mzg5NTgsMC4wNzQ1MDUxMDk2IEM0LjQ0NTI2OTM3LDAuMDc0NTA1MTA5NiA1LjcxNDAxMjA2LDEuMzYyMTk0NTggNS43MTQwMTIwNiwyLjkwMTgyMzI5IFoiIC8+PC9nPjwvc3ZnPg==)](https://orcid.org/0000-0003-0006-0188 "ORCID 0000-0003-0006-0188")
Affiliation: Instituto de Alta Investigación, Universidad de Tarapacá,
Casilla 7D, Arica, 1010000, Chile

Received November 22, 2025

###### Abstract

We develop a text-to-SQL (structured query language) system based on
large language models (LLMs) using in-context learning and apply it to
the Automatic Learning for the Rapid Classification of Events (ALeRCE)
astronomical database. ALeRCE is a community broker for the Zwicky
Transient Facility and the Vera C. Rubin Observatory. The system enables
users to query the database in natural language (NL) and generates
executable SQL queries. To develop and evaluate the system, we
constructed a dataset of 110 NL/SQL pairs. We propose a step-by-step
generation framework comprising four modules: schema linking, query
classification, prompt decomposition, and self-correction. The
performance of thirteen LLMs is evaluated using in-context learning and
prompt engineering techniques. Text-to-SQL performance is assessed using
the perfect-match (PM) rate for row identifiers (e.g., object
identifiers) and column identifiers (i.e., column names). The proposed
step-by-step framework consistently outperforms a direct-inference
baseline, while the self-correction module consistently reduces
execution errors. For Claude Opus 4.6, PM performance on row (column)
identifiers is high for simple queries, reaching 0.97 (0.94), and
decreases with query complexity to 0.44 (0.72) for medium queries and
0.59 (0.49) for hard queries. Among the thirteen evaluated models, the
best-performing LLMs for the text-to-SQL task are Claude Opus 4.6,
Gemini 2.5 Pro, Gemini 3 Flash, and GPT-5.2-Codex.

###### Key Words.

Astronomical data bases – Methods: data analysis – Methods: statistical

## 1 Introduction

Large Language Models (LLMs) are revolutionizing the field of natural
language processing (NLP) and its applications. These models are
pre-trained on massive amounts of text data and can be fine-tuned to
specific tasks such as language translation or question answering. In
the field of astronomy, the main applications up to now have been on the
named entity recognition (NER) task ([Grezes et al., 2024](#bib.bib25);
[Ghosh et al., 2022](#bib.bib17); [Sotnikov and Chaikova,
2023](#bib.bib15); [Shao et al., 2024](#bib.bib28)), question-answering
([Perkowski et al., 2024](#bib.bib27)), hypothesis generation ([Ciuca et
al., 2023](#bib.bib14)), text summarization ([Coughlin et al.,
2023](#bib.bib13)), and Artificial Intelligence (AI) research assistants
([Joseph et al., 2026](#bib.bib35); [Ye et al., 2025](#bib.bib34)). NER
aims to identify and classify named entities in text, such as
organizations, citations, surveys, telescopes, and celestial objects.
For example, AstroBERT ([Grezes et al., 2024](#bib.bib25)) identifies
organizations in the acknowledgment section of papers from the NASA
Astrophysics Data System (ADS) database¹¹ 1
[https://ui.adsabs.harvard.edu/](https://ui.adsabs.harvard.edu/).
Astro-mT5 ([Ghosh et al., 2022](#bib.bib17)) identifies 32 named
entities from the astrophysics literature data provided by the DEAL
SharedTask team. [Sotnikov and Chaikova (2023)](#bib.bib15) extracts
named entities from the Gamma-ray Coordinates Network (GCN) circulars,
including event ID, object name, observed event, observed object, and
physical phenomena. AstroLLaMA fine-tunes LlaMA-2 ([Nguyen et al.,
2023](#bib.bib26)), leveraging a corpus of 300,000 astronomy abstracts.
Downstream tasks include text generation (e.g., completing abstracts)
and embedding space quality (e.g., where similar abstracts can be
retrieved by computing the cosine similarity between the vector
embeddings).

AstroLLaMA-chat is an enhanced version of AstroLLaMA, focused on the
task of question answering ([Perkowski et al., 2024](#bib.bib27)).
Pathfinder ([Iyer et al., 2024](#bib.bib29)) is a machine-learning
framework for literature review and knowledge discovery in astronomy,
focused on semantic searching with natural language. It is an
open-source tool that uses a corpus of more than 300,000 abstracts from
NASA ADS. [Ciuca et al. (2023)](#bib.bib14) focused on hypothesis
generation about Galactic Astronomy using a primal LLM for in-context
learning with 1000 papers from NASA ADS, and an adversarial LLM model to
critique the idea.

Text summarization is the task of generating a shorter version of a
document that preserves the important information and key points while
reducing the length. [Coughlin et al. (2023)](#bib.bib13) developed
Skyportal, an open-source package designed to efficiently discover
interesting transients, including multimessenger features, manage
follow-up, perform characterization, and visualize results. Skyportal
uses ChatGPT to provide human-readable summaries of individual sources,
using redshift, classifications, and comments.

Recently, LLMs have been tested as AI agents used as research
assistants. ReplicationBench ([Ye et al., 2025](#bib.bib34)) is a
benchmark for evaluating whether AI agents can replicate entire research
papers in astronomy. This includes the experimental setup, derivations,
data analysis, and codebase. This is a challenging task, and the best
LLM scored under 20%. ASTROVISBENCH ([Joseph et al., 2026](#bib.bib35))
is a benchmark for scientific computing and visualization in astronomy.
An evaluation of state-of-the-art LLMs revealed a significant gap in
their ability to serve as effective research assistants.

Another relevant application of LLMs is the task of text-to-SQL (T2S)
parsing, which aims to convert a natural language (NL) question about a
database to its corresponding structured query language (SQL) query,
that must be executable. ScienceBenchmark ([Zhang et al.,
2023](#bib.bib19)) is a database containing NL questions and SQL queries
for benchmark purposes, which includes partially the Sloan Digital Sky
Survey (SDSS) database²² 2
[https://skyserver.sdss.org/](https://skyserver.sdss.org/). However, the
highest execution accuracy obtained in SDSS using LLMs reached only 33%.
The authors conclude that the proposed benchmark is highly challenging
and that the T2S task is far from being solved, especially for complex,
scientific datasets.

In this work, we develop a T2S system based on LLMs to be applied to the
database of the Automatic Learning for the Rapid Classification of
Events (ALeRCE). ALeRCE ([Förster et al., 2021](#bib.bib6);
[Carrasco-Davis et al., 2021](#bib.bib7); [Sánchez-Sáez et al.,
2021](#bib.bib5)) is a Chilean-led astronomy broker that is processing
the alert stream from the Zwicky Transient Facility ([Bellm et al.,
2019](#bib.bib32), ZTF;), and it has been selected as one of the
Community brokers for the Vera C. Rubin Observatory and its Legacy
Survey of Space and Time ([Ivezić et al., 2019](#bib.bib33), LSST;).
Currently, ALeRCE has more than 27,000 users from 139 countries³³ 3
Estimation from Google Analytics. The ALeRCE database⁴⁴ 4
[https://science.alerce.online/](https://science.alerce.online/) can be
accessed through the ALeRCE ZTF explorer⁵⁵ 5
[https://alerce.online/](https://alerce.online/), the SN hunter⁶⁶ 6
[https://snhunter.alerce.online/](https://snhunter.alerce.online/), a
ZTF API with simplified queries⁷⁷ 7
[https://api.alerce.online/ztf/v1](https://api.alerce.online/ztf/v1), or
directly via PostgreSQL.

However, producing SQL queries requires computational expertise and time
to learn a new database. The Astronomical Data Query Language (ADQL) ⁸⁸
8
[https://www.ivoa.net/documents/ADQL/20180112/](https://www.ivoa.net/documents/ADQL/20180112/)
includes astronomy-specific geometry functions, but its use has remained
highly technical. Under the idea of democratizing access to data in the
Rubin era, we aim to develop a system where the input is a question in
NL text for the ALeRCE database, and the output is the corresponding
executable SQL query. To this end, we propose a step-by-step framework
for performing the T2S parsing task using in-context learning with LLMs,
and compare its performance with a direct inference system. In Sect.
[2](#S2 "2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
we present a historical context of T2S parsing and related work. In
Sect.
[3](#S3 "3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
we describe our methodology, wherein we construct a dataset of NL
questions and their corresponding SQL queries for the ALeRCE database,
which we make publicly available. Section
[4](#S4 "4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
presents our results, where we evaluate the performance of thirteen LLMs
using in-context learning and analyze the errors. In Sect
[5](#S5 "5 Discussion ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
we discuss possible improvements such as function calling and structured
outputs. Finally, we draw conclusions in Sect.
[6](#S6 "6 Conclusions ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").

## 2 Background

### 2.1 Background on text-to-SQL parsing

The T2S task can be defined as follows ([Qin et al., 2022](#bib.bib16)):
Given a NL question Q and the corresponding database schema S=(T,C),
where T stands for tables and C for columns, the goal is to generate an
SQL query Y, that when executed will return results that match the
user’s intent ([Katsogiannis-Meimarakis and Koutrika,
2023](#bib.bib20)). The question Q is a sequence of \|Q\| tokens. The
database consists of \|T\| tables and \|C\| columns. Each table is
described by its name, which contains multiple words. Each column within
a table is also described by words. Schema linking, which consists of
aligning entities in questions to tables or columns is the most
important topic of T2S problems ([Li et al., 2024](#bib.bib12)). In
addition, external knowledge can be used to improve the model’s
comprehension of the problem. Typically, this refers to domain-specific
knowledge, e.g., astronomy, as well as mathematical computation. The T2S
task presents two kinds of challenges. The first is related to
understanding the NL question. The NL is inherently ambiguous
([Katsogiannis-Meimarakis and Koutrika, 2023](#bib.bib20)), i.e., the
formulation of expressions is open to more than one interpretation. The
second is that SQL has a strict syntax, which leads to limited
expressivity compared to NL. This makes building the syntactically and
semantically correct SQL query difficult based on the underlying
database schema.

For evaluating T2S systems, several database benchmarks have been
created, such as Spider ([Yu et al., 2018](#bib.bib4)), BIRD ([Li et
al., 2024](#bib.bib12)), and ScienceBenchmark ([Zhang et al.,
2023](#bib.bib19)). The Spider dataset contains 10,181 NL questions and
5,693 unique SQL queries over 200 databases belonging to 139 different
domains, allowing it to deal with the cross-domain T2S parsing task
([Qin et al., 2022](#bib.bib16)). The queries are divided into four
levels of difficulty: easy, medium, hard, and extra hard. However,
Spider mainly contains simple databases with few tables, columns, and
entries, which rarely reflect real-world problems ([Zhang et al.,
2023](#bib.bib19)). BIRD is a large-scale cross-domain benchmark
covering 95 databases from 37 domains. [Li et al. (2024)](#bib.bib12)
performed an error analysis using GPT-3.5 Turbo in the T2S task on BIRD.
The authors found that the most common error is wrong schema linking
(41.6%), i.e., erroneous association of tables and columns to the NL
question. The second most common error was misunderstanding database
content (40.8%), i.e., failing to recall the correct database structure
(e.g., using a column that does not belong to a specific table) or
generating fake schema items (e.g., calling a table or column that does
not belong to the database). However, the previous two database
benchmarks do not contain domain-specific knowledge of astronomy.
ScienceBenchmark is a new database benchmark that contains three
real-world and domain-specific databases: research policy-making,
astrophysics, and cancer research. A subset of the SDSS database is used
(5 out of 10 tables) due to limitations in the number of tokens allowed
by LLMs. For this database, 200 NL/SQL pairs were manually generated by
a team of domain and SQL experts, and data augmentation was carried out.
The authors found that state-of-the-art T2S algorithms that obtained
over 80% of execution accuracy on Spider, achieved only 21% execution
accuracy on SDSS. In addition, GPT-3.5 with prompting achieved only 33%
execution accuracy. This poor performance is presumably because SDSS is
a domain-specific database that includes many numerical values and
requires the use of functions and mathematical operators. In addition,
it contains many column names and values labeled with domain-specific
abbreviations, e.g., z for redshift. The authors concluded that the
current state-of-the-art approaches do not perform well on real-world
problems, and that we need domain-specific benchmarks for training and
evaluating T2S systems in scientific domains.

### 2.2 Prompt engineering for the T2S task

LLMs are currently language models with billions of parameters, trained
on massive amounts of text data. According to [Zhao et al.
(2026)](#bib.bib2), LLMs show emergent abilities such as in-context
learning, instruction following, and step-by-step reasoning. In-context
learning allows LLMs to convert a NL question into a SQL query using a
prompt text, i.e., it is an inference task (there is no training or
fine-tuning). The prompt usually includes four components: a task
instruction, a NL question, the database schema, and external knowledge
([Chang and Fosler-Lussier, 2023](#bib.bib1)). Figure
[1](#S2.F1 "Figure 1 ‣ 2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the four components of a basic prompt for the T2S task, which is
the basic inference task using in-context learning, and serves as a
baseline for comparison purposes. We call this scheme Direct Query
Generation.

Figure 1: Four components of a basic prompt for the T2S task. This
approach is called direct query generation.

Several studies have shown that the ability of LLMs for complex
reasoning can be improved using step-by-step reasoning ([Kojima et al.,
2022](#bib.bib3)). There are several prompting techniques for multi-step
reasoning, such as Chain of Thought ([Wei et al., 2022](#bib.bib8),
CoT;), which is able to guide LLMs through a logical reasoning chain,
mimicking how humans break down problems into logical intermediate steps
([Sahoo et al., 2024](#bib.bib21)). A variant is the Logical
Chain-of-Thought ([Liu et al., 2023](#bib.bib22)) prompting, which
includes effective verification mechanisms to reduce logical errors and
hallucinations through a think-verify-revise loop ([Zhao et al.,
2024](#bib.bib10)). Regarding code generation, Structured
Chain-of-Thought (SCoT) prompting incorporates program structures
(sequence, branch, and loop structures) into reasoning steps ([Li et
al., 2025](#bib.bib9)).

The DIN (Decomposed In-Context)-SQL method ([Pourreza and Rafiei,
2023](#bib.bib11)) decomposes the text-to-SQL task into four modules: 1)
schema linking, 2) query classification and decomposition, 3) SQL
generation, and 4) self-correction. A prompt-based module is designed
for schema linking, which includes ten random samples from the training
set of the Spider dataset. For each mention of a column name in the
question, the corresponding column and its table are selected from the
given database schema. The second module classifies each query into one
of three classes: easy, non-nested complex, and nested complex. The
class labels are essential for the query generation module, which uses
different prompts for each query class. A simple few-shot prompting with
no intermediate steps is performed for questions in the easy class. In
addition to class labels, the module also detects the tables to be
joined for non-nested and nested queries. The nested complex class is
the hardest and requires several intermediate steps before generating
the final answer. The prompt for this class is designed in a way that
the LLM first solves sub-queries and then uses them to generate the
final answer. Finally, a self-correction module is added, where only
buggy code is provided to the model, and it is asked to fix the bugs in
a zero-shot setting.

The DAIL-SQL ([Gao et al., 2024](#bib.bib18)) proposes a new prompt
engineering method focused on question representation and in-context
learning for text-to-SQL. Among the question representations studied are
a basic prompt (table schema and question in NL), text-representation
prompt (add task instruction to the basic prompt), OpenAI Demonstration
Prompt (ODp) (the instruction is more specific, constraining how the
model must answer), and Code Representation Prompt (CRp) (it presents
the T2S task in SQL syntax, e.g., the NL questions appear inside SQL
comments). Regarding in-context learning, DAIL-SQL focuses on example
selection and example organization. Among the example selection
strategies are random selection, question-similarity selection (select k
examples with the most similar questions), query-similarity selection
(select k examples similar to the target SQL query), and DAIL Selection
(consider both questions and queries to select examples). Experiments
were run with the Spider and Spider-Realistic datasets. In the zero-shot
scenario, the ODp representation performed best in GPT-3.5 Turbo, but
the basic prompt performed best in GPT-4. Adding foreign key
information, i.e., the relationships between tables in a relational
database, significantly improves the execution accuracy of LLMs. Adding
the rule to generate SQL queries “with no explanation” consistently
improved the performance of all LLMs. DAIL selection generally
outperformed other example selection strategies. Similar approaches to
DIN-SQL and DAIL-SQL have been presented in MAC-SQL ([Wang et al.,
2025](#bib.bib23)), and C3 ([Dong et al., 2023](#bib.bib24)).

## 3 Methodology

### 3.1 ALeRCE database

As of early 2026, the ALeRCE database has 25 tables and 304 columns; see
the entity-relationship diagram in Appendix
[A](#A1 "Appendix A ALeRCE database ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
It contains information about astrophysical objects, including global
and band-dependent statistics, time- and band-dependent flux evolution,
cross-matches, and machine-learning probabilities. For example, the
ALeRCE light curve classifier ([Sánchez-Sáez et al., 2021](#bib.bib5))
computes machine learning probabilities for 15 classes: SNIa, SNIbc,
SNII, SLSN, QSO, AGN, Blazar, CV/Nova, YSO, LPV, EB, DSCT, RRL, CEP, and
Periodic-Other. The main tables are ‘object’ (22 columns), ‘probability’
(6 columns), ‘magstat’ (28 columns), ‘detection’ (30 columns),
‘non_detection’ (4 columns), ’forced_photometry’ (42 columns), and
‘feature’ (5 columns). These tables contain information about each
object, including time and spectral band statistics. The organization of
the database is centered on the ‘object’ and ‘probability’ tables, which
contain the attributes and classification of the objects, respectively,
and which are requested in most of the queries requiring JOIN⁹⁹ 9
Joining in SQL means retrieving data from two or more tables based on a
common field. commands or sub-queries. To collect NL/SQL pairs, we first
included 21 examples generated by the ALeRCE team and presented in
workshops with the aim of teaching users how to interact with the
database. In addition, a group of 10 astronomers were tasked with
generating real-world questions, with the only restriction being that
the information to answer these questions must be obtainable from the
ALeRCE database alone. In this way, another 89 examples were generated.
Finally, an SQL expert built the target gold queries, resulting in a
total of 110 NL/SQL pairs obtained. Listing
[1](#LST1 "Listing 1 ‣ 3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows an example of a query, where ranking=1 means to take the highest
probability only.

[⬇](data:text/plain;base64,PEBcdGV4dGNvbG9ye3JlZH17VXNlciBSZXF1ZXN0fUA+Ogo8QFx0ZXh0Y29sb3J7YmxhY2t9e0dpdmUgbWUgdGhlIG9iamVjdHMgY2xhc3NpZmllZCBhcyBZU08gYnkgdGhlaXJ9QD4KPEBcdGV4dGNvbG9ye2JsYWNrfXtsaWdodGN1cnZlcyAod2l0aCBhIHByb2JhYmlsaXR5IGhpZ2hlciB0aGFuIDAuNyl9QD4KPEBcdGV4dGNvbG9ye2dyZWVufXtRdWVyeX1APjoKU0VMRUNUCiAgICBvaWQsIHByb2JhYmlsaXR5CkZST00KICAgIHByb2JhYmlsaXR5CldIRVJFCiAgICBjbGFzc2lmaWVyX25hbWU9J2xjX2NsYXNzaWZpZXInIEFORAogICAgY2xhc3NfbmFtZT0nWVNPJyBBTkQKICAgIHJhbmtpbmc9MSBBTkQKICAgIHByb2JhYmlsaXR5ID4gMC43)

User Request:

Give me the objects classified as YSO by their

lightcurves (with a probability higher than 0.7)

Query:

SELECT

oid, probability

FROM

probability

WHERE

classifier_name=’lc_classifier’ AND

class_name=’YSO’ AND

ranking=1 AND

probability \> 0.7

Listing 1: Example of a question in natural language and its
corresponding SQL query.

Most of the SQL queries included in our gold set represent typical
requests found in the literature, which show how the community has
already used ALeRCE. These include: requesting light curve detections
for given objects ([Magee et al., 2023](#bib.bib36); [Gomez and Gezari,
2023](#bib.bib37); [Müller-Bravo et al., 2025](#bib.bib39); [Alexander
et al., 2026](#bib.bib38), e.g.,); radial queries ([Pomeroy and Norris,
2024](#bib.bib40), e.g.,); crossmatches with objects from a given class
in one of the ALeRCE classifiers ([López-Navas et al.,
2022](#bib.bib42); [Arévalo et al., 2024](#bib.bib43); [Oshikiri et al.,
2024](#bib.bib41), e.g.,). Meanwhile, other queries are highly specific,
including e.g., queries to individual features, or filters through
tables other than object and probability (data quality, the Panoramic
Survey Telescope & Rapid Response System (PS1)¹⁰¹⁰ 10
[https://catalogs.mast.stsci.edu/panstarrs](https://catalogs.mast.stsci.edu/panstarrs)
data, Wide-field Infrared Survey Explorer (WISE)¹¹¹¹ 11
[https://irsa.ipac.caltech.edu/Missions/wise.html](https://irsa.ipac.caltech.edu/Missions/wise.html)
data, and Solar System data); while these requests are expected to be
less frequent, they allow us to clean object samples and/or get a
broader understanding of the selected objects, and thus deserve to be
included in our gold set.

### 3.2 Query classification

Queries were classified into three difficulty levels: simple, medium,
and hard. This was done first to evaluate how the LLM performance varies
with query difficulty and, second, to allow generating a different
prompt depending on the level of difficulty, as explained in Sect.
[3.5](#S3.SS5 "3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
To this end, we define the tables ‘object’, ‘probability’, and ‘magstat’
as basic tables (the most commonly used) and the remaining tables as
general. Definitions of the categories are as follows:

- •
  Simple: queries using up to two basic tables, or queries using a
  single general table.
- •
  Medium: queries that require either one general and one basic table,
  two general tables, or three basic tables. This category also includes
  queries that contain two simple sub-queries—each comparable in
  complexity to a simple query—or one sub-query of higher complexity.
- •
  Hard: Any query requiring more than three tables or using more than
  one non-simple sub-query for optimization.

Based on these, the ALeRCE dataset of 110 NL/SQL pairs was partitioned
into a development set of 58 examples and a testing set of 52 examples.
Note that when using in-context learning, there is no need for a
training set, since there is no parameter adjustment. The development
set is used for developing and evaluating diverse prompt methods. The
testing set is exclusively used for the final evaluation. The 58
examples of the development set consisted of 32 simple, 14 medium, and
12 hard queries. The 52 examples of the testing set were composed of 32
simple, 10 medium, and 10 hard queries. Examples of simple, medium and
hard queries are given in Appendix
[B](#A2 "Appendix B Examples of queries ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").

### 3.3 LLM models

Table
[1](#S3.T1 "Table 1 ‣ 3.3 LLM models ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
describes the thirteen LLM models used in this work. This includes the
full name of the API, the abbreviations of names used in what follows,
and the context length. The latter is the maximum number of tokens
(input + output) that an LLM can process in a single forward pass.

| API Model                     | Abbreviation      | Context length |
| ----------------------------- | ----------------- | -------------- |
| gpt-4o-2024-11-20             | GPT-4o            | 128 000        |
| gpt-4.1-2025-04-14            | GPT-4.1           | 1 048 576      |
| gpt-5-2025-08-07              | GPT-5             | 400 000        |
| gpt-5.2-2025-12-11            | GPT-5.2           | 400 000        |
| gpt-5.2-codex                 | GPT-5.2-Codex     | 400 000        |
| gpt-5.3-codex                 | GPT-5.3-Codex     | 400 000        |
| claude-3-7-sonnet-20250219    | Claude 3.7        | 200 000        |
| claude-sonnet-4-5-20250929    | Claude Sonnet 4.5 | 200 000        |
| claude-opus-4-6               | Claude Opus 4.6   | 200 000        |
| gemini-2.5 flash              | Gemini 2.5 Flash  | 1 048 576      |
| gemini-2.5 pro                | Gemini 2.5 Pro    | 1 048 576      |
| gemini-3-flash-preview        | Gemini 3 Flash    | 1 048 576      |
| gemini-3.1-flash-lite-preview | Gemini 3.1 Flash  | 1 048 576      |

Table 1: Description of the LLM models.

### 3.4 Basic prompt

The basic inference model shown in Fig.
[1](#S2.F1 "Figure 1 ‣ 2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
is used as a baseline for comparison purposes. In the zero-shot case,
the information given to the model includes the requested T2S task, the
database schema, external knowledge (which is query-dependent), and
general information about the database structure. The format of the
zero-shot prompt is shown in Listing
[2](#LST2 "Listing 2 ‣ 3.4 Basic prompt ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
Optionally, a few examples could be added to the prompt, as in the
so-called few-shot case. In this work, we mainly deal with the zero-shot
case. The few-shot case is presented in Sect.
[5](#S5 "5 Discussion ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
for the best LLM models.

[⬇](data:text/plain;base64,e0dlbmVyYWxfVGFza30KCiMgQ29udGV4dDoKe0dlbmVyYWxfQ29udGV4dH0KCiMgVGhlIERhdGFiYXNlIGhhcyB0aGUgZm9sbG93aW5nIHRhYmxlcyB0aGF0IGNhbiBiZSB1c2VkIHRvIGdlbmVyYXRlIHRoZSBTUUwgcXVlcnk6Cnt0YWJsZXNfc2NoZW1hfQoKe0ZpbmFsX0luc3RydWN0aW9uc30KCiMgSW1wb3J0YW50IEluZm9ybWF0aW9uIGZvciB0aGUgcXVlcnkKe0V4dGVybmFsX0tub3dsZWRnZX0KIyBVc2VyIFJlcXVlc3Q6ICcne3F1ZXJ5fScn)

{General_Task}

\# Context:

{General_Context}

\# The Database has the following tables that can be used to generate
the SQL query:

{tables_schema}

{Final_Instructions}

\# Important Information for the query

{External_Knowledge}

\# User Request: ’’{query}’’

Listing 2: Zero-shot prompt format.

### 3.5 Proposed framework for the T2S task

Our end goal is to develop a practical virtual assistant for ALeRCE
users. With this aim, we decompose the prompt into different steps that
run autonomously. Our approach is inspired by previous work such as
DIN-SQL ([Pourreza and Rafiei, 2023](#bib.bib11)) and DAIL-SQL ([Gao et
al., 2024](#bib.bib18)). The proposed framework for performing the T2S
task in the ALeRCE database has four steps: schema linking,
classification, decomposition, and self-correction. The first step,
schema linking, is shown in Fig.
[2](#S3.F2 "Figure 2 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
Steps 2 and 3 are shown in Fig.
[3](#S3.F3 "Figure 3 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
Finally, Fig.
[4](#S3.F4 "Figure 4 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the self-correction step, the last step of the proposed framework.
A detailed description of the four steps of the proposed framework is as
follows:

1.  1.  Schema Linking: The aim is to align the question with the database
        schema, i.e., identify the required tables and columns. With this
        aim, an independent module was designed to extract the required
        structure from the database; see Fig.
        [2](#S3.F2 "Figure 2 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
        This is carried out in two stages. The first identifies the required
        tables, and the second determines the columns. Each stage is
        performed using zero-shot prompting. For the task of extracting the
        tables, the model is provided with a question, a description of the
        tables in the database, and it is instructed to perform the schema
        linking task. The columns are extracted independently for each of
        the previously selected tables. A different API call is started for
        each table, with a prompt describing the columns contained in the
        table. The outputs of this step are the selected schema tables. The
        prompt for the schema linking is shown in the Appendix
        [C](#A3 "Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
        Listing
        [C.1](#LST1a "Listing C.1 ‣ Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
2.  2.  Classification: This step determines the level of difficulty of the
        query. The prompt for classifying the level of difficulty uses the
        structure shown in the Appendix
        [C](#A3 "Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
        Listing
        [C.2](#LST2a "Listing C.2 ‣ Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
        The prompt includes the definition of the query classes by
        difficulty, the tables and columns obtained in the schema linking
        stage, and the user query.
3.  3.  Decomposition: As shown in Fig.
        [3](#S3.F3 "Figure 3 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
        a decomposition prompt is performed for the medium and hard queries.
        For simple queries, decomposition is not needed; they go directly to
        the generation prompt block. Otherwise, we decompose the problem
        into sub-problems. This is useful for complex queries that require
        different sub-queries. We tested least-to-most prompting ([Zhou et
        al., 2023](#bib.bib31)) and decomposed prompting ([Khot et al.,
        2023](#bib.bib30)). A new module is used to decompose the queries
        that require complex sub-queries or JOINs. The input to this module
        is the information extracted in the previous stages.
4.  4.  Self-Correction: Recent LLM models, such as GPT-5.2-Codex, Claude
        Opus 4.6 or Gemini 3 Flash, are able to correct or improve their
        original response. A new agent can be used to get feedback about
        previous responses. The self-correction module addresses the three
        main types of execution errors found in the generation of queries:
        timeout; non-existing structures, such as confusion of column and
        table names, or non-existent columns, tables, functions, etc.; and
        general errors such as syntax, missing FROM-clauses, and so on. A
        schema of the self-correction step is shown in Fig.
        [4](#S3.F4 "Figure 4 ‣ 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
        The self-correction step consists of a single iteration and is
        triggered only when the generated SQL query produces an execution
        error, so it is not applied to every query. This step is
        particularly effective at fixing syntax errors and undefined or
        ambiguous references. Timeout errors can also be corrected when the
        model identifies the minimal conditions and clauses needed to
        produce a valid result within the time limit. A different zero-shot
        prompting is used for each type of error. For the timeout error
        (more than 2 minutes), it is assumed that there is an optimization
        problem. The agent is asked to improve or re-order the query
        according to the error found. In addition, advice on avoiding this
        type of error is provided. The main structure of the prompt for
        timeout errors is shown in Appendix
        [C](#A3 "Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
        Listing
        [C.3](#LST3 "Listing C.3 ‣ Appendix C Prompt formats ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
        A similar structure is used for general errors and non-existent
        structures, but focusing the prompt on the syntax and the database
        structure errors.

Figure 2: Diagram of the schema linking, the first step of the proposed
framework for the T2S task.

Figure 3: Step-by-step query generation approach includes a
classification prompt and a decomposition prompt. The classification
prompt aims to discriminate between simple, medium, and hard queries.
The decomposition prompt divides the problem into sub-problems for
medium and hard queries. Simple queries skip the decomposition and go
directly to the generation prompt.

Figure 4: Self-correction step generates a prompt specific to each type
of SQL error: time-out, non-existing structures, or general errors.

### 3.6 Prompt engineering

The prompt is separated into system messages and user messages, as done
in OpenAI¹²¹² 12
[https://platform.openai.com/docs/guides/prompt-engineering/#messages-and-roles](https://platform.openai.com/docs/guides/prompt-engineering/#messages-and-roles).
The system messages include the general task, the ALeRCE context, and
the database schema. The user messages include the user’s question and
the external knowledge, if needed. For the general task, a personality
as a data scientist expert is given:

> “As an SQL expert with a willingness to assist users…”

The general context describes the ALeRCE pipeline, the objects and
classes used, as well as details on how ALeRCE works (including the
probability assigned to objects). The following is an example:

> “ALeRCE Pipeline Details
>
> - Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related
>   to new objects undergo stamp-based classification.
> - Light Curve Classifier (denoted as ‘lc_classifier’): A balanced
>   hierarchical random forest classifier employing four models and 15
>   classes.  
>   …”

For the database schema, commands to create the ALeRCE SQL tables are
given:

> “CREATE TABLE object (/\* this is the most important table. It
> contains the main statistics of an object, independent of time and
> band \*/  
> oid VARCHAR PRIMARY KEY  
> …)”

In addition, instructions on the format of the answers are given:

> “Answer ONLY with the SQL query …”, “… Do NOT CHANGE the names of the
> tables or columns unless the user explicitly asks you to do …”

For the step-by-step method, different prompts were generated for each
step. For the schema linking step, the prompt included a brief
description of the task, a description of the tables of the database,
and specifications regarding when to select the object, probability, and
feature tables. This is because the object and probability tables are
designed for frequent use and have been indexed accordingly, while the
features table is designed for more advanced use cases and is generally
more expensive to query. For the classification step, a prompt with the
classification task, a description of the three query categories, the
table schema selected from the schema linking step (including column
descriptions), instructions associated with the classification, and the
user request are used.

For the decomposition step, the query generation was split into two
stages. In the first stage, the LLM is asked to generate a step-by-step
plan to generate the query, including all requirements, conditions, and
details. In the second stage, the LLM is requested to generate the query
following the plan generated in the first stage. For medium queries, a
prompt example for the first stage is the following:

> “Your task is to DECOMPOSE the user request into a series of steps
> required to generate a PostgreSQL query …”.

For hard queries, a higher emphasis is put on the details and order of
the query structure, as follows:

> “The request is a very difficult and advanced query, so you will need
> to use JOIN, INTERSECT, and UNION statements, together with Nested
> queries. It is very important that you give every possible detail in
> each step, …”.

Following BIRD ([Li et al., 2024](#bib.bib12)), we add external
knowledge to the prompts in order to provide the information needed to
solve the problem, but that is not present in the dataset, such as
numeric reasoning knowledge, domain knowledge, and synonym knowledge.
This external knowledge is specific to each query. The following is an
example of the external knowledge required to solve a simple query:

> Request: “Query all objects that were first classified as SN by the
> stamp classifier between August 17 and August 21, 2024, with a
> probability greater than 0.5 or at least two detections.”  
> External Knowledge:  
> – MJD date for August 17 2024 = 60173.0  
> – MJD date for August 21 2024 = 60177.0

Notice that we used the same prompts for all LLMs, following established
prompt engineering practices. This independence of the LLM provider
allows us to make fair comparisons and to use these prompts with new
versions of the LLMs, when the old models become deprecated. In
addition, it allows the users to select an LLM of their own choice.

### 3.7 Evaluation metrics

The most common metric used in T2S is the execution accuracy (EX). EX is
the proportion of questions in the test set for which the execution
results of both the predicted and ground-truth inquiries are identical
([Li et al., 2024](#bib.bib12)). In this work, we propose a modified EX,
to consider partially correct results. We compute the set of row
identifiers, called ids, in the ground truth, $`O_{t}`$, as well as the
set of the predicted ids, $`O_{p}`$. In the ALeRCE database, examples of
row identifiers are ‘object identifier’ (oid), candidate identifier
(candid), oid_catalog, objectidps1 (object id closest source from PS1
catalog), classifier_name, or count.

For this binary classification problem, precision and recall are
calculated. For each predicted query ($`k`$, $`l`$), where $`k`$ is the
user query identifier and $`l`$ is the query run identifier (e.g., an
index from 0 to 9 for 10 runs of the user query $`k`$), the recall
metric is calculated by counting the predicted identifiers
($`id_{i}^{k,l}`$, where $`i`$ is the row number among the predicted
values) that match the set of true ids ($`O^{k}_{t}`$), divided by the
cardinality of the set of true ids, $`|O^{k}_{t}|`$:

|     |     |     |     |
| --- | --- | --- | --- |
|     |

````math
r_{k,l}=\frac{\sum_{i\in\text{pred}}\mathit{score}(id_{i}^{k,l},O^{k}_{t})}{|O^{k}_{t}|},
``` |  | (1) |

where

|     |                                           |     |     |
|-----|-------------------------------------------|-----|-----|
|     |
       ``` math
       \mathit{score}(id,O)=\cases{1},&id\in O\\
       0,id\notin O{}\lx@close@alignment.
       ```                                        |     | (2) |

Likewise, for each user query $`k`$ and query run $`l`$, the precision
metric is calculated by counting the matches between the true ids
($`id_{i}^{k,l}`$, where $`i`$ is the row number among the true values)
and the set of predicted ids ($`O^{k,l}_{p}`$), and dividing this amount
by the cardinality of the set of predicted ids, $`|O^{k,l}_{p}|`$:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
p_{k,l}=\frac{\sum_{i\in\text{true}}\textit{score}({id}^{k,l}_{i},O^{k,l}_{p})}{|O^{k,l}_{p}|}.
``` |  | (3) |

The same can be done in the column dimension, where, for example, an oid
can have associated values such as firstmjd, lastmjd, ndet (number of
detections), and so on. Thus, we define $`r_{k,l}^{rows}`$ and
$`p_{k,l}^{rows}`$ as the recall and precision computed along the row
identifiers. Likewise $`r_{k,l}^{cols}`$ and $`p_{k,l}^{cols}`$ are the
recall and precision computed along the column identifiers.

In this work, we assume that an answer containing all and only the
expected rows is correct. On the other hand, we assume that an answer
containing all but not necessarily only the expected columns is correct.
This is because the latter will not significantly impact the user
experience. According to this assumption, we evaluate all prompting
methods using a metric of perfect matching rates ($`PM`$), defined as
follows:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
PM=\frac{1}{n_{\rm queries}\penalty\ n_{\rm runs}}\sum_{k\in{\rm queries}}\sum_{l\in{\rm runs}}N_{\rm perfect}(k,l),
``` |  | (4) |

where $`n_{\rm queries}`$ is the total number of queries, and
$`n_{\rm runs}`$ is the number of runs per query, typically 10.
$`N_{\rm perfect}(k,l)`$ is defined as:

|  |  |  |  |
|----|----|----|----|
|  |
``` math
N^{\rm rows}_{\rm perfect}(k,l)=\cases{1},\text{ if $p^{\rm rows}_{k,l}$ = $r^{\rm rows}_{k,l}$ = 1}\\
0,\text{ otherwise}{}\lx@close@alignment,
``` |  | (5) |

|  |  |  |  |
|----|----|----|----|
|  |
``` math
N^{\rm cols}_{\rm perfect}(k,l)=\cases{1},\text{ if $r^{\rm cols}_{k,l}$ = 1}\\
0,\text{ otherwise}{}\lx@close@alignment,
``` |  | (6) |

for the case of rows and columns, respectively. Accordingly, we define
$`PM_{\rm rows}`$ and $`PM_{\rm cols}`$ as metrics for perfect matches
among the rows or columns, respectively.

## 4 Results

In this section, we present results comparing the performance of direct
vs. step-by-step (sbs) query generation methods, with and without
self-correction, as a function of query difficulty across the thirteen
LLMs. In addition, we analyze the errors obtained. Table
[21](#footnote21 "footnote 21 ‣ Table 9 ‣ Appendix D Perfect match rates by LLM ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
in Appendix
[D](#A4 "Appendix D Perfect match rates by LLM ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the PM results. For evaluation, we conduct ten runs and compute
the average PM across the entire query set. We report the mean and
standard deviation across these runs. Consistency varies across LLM
models, even when using a zero temperature. For earlier models, such as
Claude 3.7, GPT-4o and GPT-4.1, SQL outputs vary more noticeably for
medium and hard queries, where more tokens are required and thus the
probability of divergence increases (see the standard deviation in Table
[21](#footnote21 "footnote 21 ‣ Table 9 ‣ Appendix D Perfect match rates by LLM ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")).
Newer models have more deterministic behavior, notably Gemini 3 Flash,
Claude Sonnet 4.5, and Claude Opus 4.6. It can be observed in Table
[21](#footnote21 "footnote 21 ‣ Table 9 ‣ Appendix D Perfect match rates by LLM ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
that when comparing the performance of each LLM independently, without
(w/o) and with (w/) self-correction, in all cases, the self-correction
improves or achieves a similar result to that without self-correction.

Fig.
[5](#S4.F5 "Figure 5 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the performance of Claude Opus 4.6 w/ and w/o self-correction at
different levels of query difficulty. When applying self-correction, the
results improve or remain the same for all kinds of queries for both the
direct query generation and the sbs method. For example, the performance
of row ids improves from 0.84 (w/o self-correction) to 0.94 (w/
self-correction) when using the direct query generation method, and from
0.94 to 0.97 when using the sbs query generation method.

![Refer to caption](2606.18108v1/results_nosc_vs_sc_v2_claude46opus.png)

Figure 5: Claude Opus 4.6 perfect matching performance for rows and
columns, with and without self-correction, for the direct and
step-by-step query generation methods.

Table
[13](#footnote13 "footnote 13 ‣ Table 2 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows a general statistical ranking of the top six LLMs according to
their PM performance, using both the sbs and direct query generation
methods with self-correction. The top six LLMs in our experiments are:
Claude Opus 4.6, Claude Sonnet 4.5, Gemini 2.5 Pro, Gemini 3 Flash,
GPT-5.2-Codex and GPT-5.3-Codex. We used an unpaired permutation test to
rank the LLMs according to their performance on rows (R) and columns
(C), for simple (S), medium (M), and hard (H) queries. A rank of 1 in
the table denotes the best-performing LLM. This rank may be assigned to
multiple models when pairwise performance differences are not
statistically significant. Next, we aggregated the results to obtain a
final sum, where a lower number indicates a better ranking. It can be
observed that the sbs query generation methods obtained a better ranking
than the direct query generation methods in all cases, except for Gemini
3 Flash, which direct query generation version ranked third. Claude Opus
4.6 (sbs) and Gemini 2.5 Pro (sbs) are ranked first and second,
respectively.

Table
[14](#footnote14 "footnote 14 ‣ Table 3 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the average total cost per run of the whole test set in US
dollars, as well as its split into the different stage components, for
the top six LLMs. Using the cost of Gemini 2.5 Pro (sbs) as a reference,
the cost increases 2.41 times when using Claude Opus 4.6 (sbs), and 1.01
times when using GPT 5.2 Codex (sbs). Gemini 3 Flash has the lowest cost
for both the direct and sbs query generation methods. Table
[10](#A5.T10 "Table 10 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
in Appendix
[E](#A5 "Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the maximum total token consumption, broken down by pipeline
stage, LLM model and query generation method. Importantly, all stages
remain well within the context window limits of the thirteen evaluated
LLMs.

| Model (Method)             | RS  | RM  | RH  | CS  | CM  | CH  | R-SUM | C-SUM | SUM |
|----------------------------|-----|-----|-----|-----|-----|-----|-------|-------|-----|
| Claude Opus 4.6 (SbS)      | 1   | 1   | 1   | 1   | 3   | 3   | 3     | 7     | 10  |
| Gemini 2.5 Pro (SbS)       | 3   | 1   | 3   | 2   | 2   | 2   | 7     | 6     | 13  |
| Gemini 3 Flash (Direct)    | 2   | 2   | 2   | 4   | 2   | 1   | 6     | 7     | 13  |
| GPT-5.2-Codex (SbS)        | 3   | 1   | 4   | 2   | 2   | 2   | 8     | 6     | 14  |
| Gemini 3 Flash (SbS)       | 2   | 2   | 3   | 2   | 2   | 3   | 7     | 7     | 14  |
| Claude Sonnet 4.5 (SbS)    | 2   | 2   | 4   | 3   | 1   | 3   | 8     | 7     | 15  |
| GPT-5.3-Codex (SbS)        | 3   | 2   | 3   | 2   | 3   | 2   | 8     | 7     | 15  |
| Claude Opus 4.6 (Direct)   | 2   | 2   | 2   | 4   | 2   | 4   | 6     | 10    | 16  |
| Gemini 2.5 Pro (Direct)    | 3   | 2   | 3   | 4   | 2   | 3   | 8     | 9     | 17  |
| GPT-5.2-Codex (Direct)     | 4   | 2   | 4   | 4   | 2   | 2   | 10    | 8     | 18  |
| Claude Sonnet 4.5 (Direct) | 3   | 1   | 5   | 5   | 3   | 3   | 9     | 11    | 20  |
| GPT-5.3-Codex (Direct)     | 5   | 3   | 4   | 5   | 4   | 2   | 12    | 11    | 23  |

Table 2: General statistical ranking (w/ self-correction).¹³¹³ 13 Notes.
An unpaired permutation test on per-run perfect-match scores
(Holm-Bonferroni correction, $`\alpha=0.05`$) was performed.

| LLM (Query Generation Method) | Total | Schema | Difficulty | Decomp | SQL | Self-Correction | Ratio |
|----|----|----|----|----|----|----|----|
| Claude Opus 4.6 (Step-by-Step) | 2.87 | 0.23 | 0.43 | 1.07 | 1.05 | 0.09 | 2.41 |
| Claude Opus 4.6 (Direct) | 1.55 | 0.23 | — | — | 1.19 | 0.13 | 1.30 |
| GPT-5.2-Codex (Step-by-Step) | 1.20 | 0.15 | 0.15 | 0.43 | 0.38 | 0.10 | 1.01 |
| Gemini 2.5 Pro (Step-by-Step) | 1.19 | 0.43 | 0.11 | 0.29 | 0.30 | 0.06 | 1.00 |
| Gemini 2.5 Pro (Direct) | 0.85 | 0.43 | — | — | 0.34 | 0.08 | 0.72 |
| GPT-5.2-Codex (Direct) | 0.82 | 0.15 | — | — | 0.56 | 0.11 | 0.69 |
| Claude Sonnet 4.5 (Step-by-Step) | 0.71 | 0.14 | 0.06 | 0.26 | 0.14 | 0.11 | 0.60 |
| GPT-5.3-Codex (Step-by-Step) | 0.61 | 0.07 | 0.07 | 0.19 | 0.21 | 0.06 | 0.51 |
| Claude Sonnet 4.5 (Direct) | 0.39 | 0.14 | — | — | 0.16 | 0.08 | 0.33 |
| GPT-5.3-Codex (Direct) | 0.34 | 0.07 | — | — | 0.20 | 0.07 | 0.28 |
| Gemini 3 Flash (Direct) | 0.30 | 0.08 | — | — | 0.20 | 0.02 | 0.25 |
| Gemini 3 Flash (Step-by-Step) | 0.27 | 0.08 | 0.04 | 0.06 | 0.08 | 0.01 | 0.23 |

Table 3: Average total cost per run in US dollars by LLM and query
generation method.¹⁴¹⁴ 14 Notes. Values are aggregated over all queries
from the test set and averaged over 10 runs. The cost is split into each
stage of the query. The ratio compares the total cost against Gemini 2.5
Pro (sbs).

Fig.
[6](#S4.F6 "Figure 6 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the performance of both query generation methods with
self-correction using Gemini 2.5 Pro, measured as the rate of PM queries
for row and column identifiers, as a function of query difficulty. It
can be observed that the proposed sbs framework obtained higher mean
values than the direct method in all cases, but there are no
statistically significant differences according to the permutation
test¹⁵¹⁵ 15
[https://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/](https://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/),
except for medium queries (rows) and simple queries (columns).

The right side of Fig.
[5](#S4.F5 "Figure 5 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the same experiment with Claude Opus 4.6. It can be observed that
the proposed sbs framework outperforms the direct method for simple and
hard queries. The differences are statistically significant according to
the permutation test, except for medium queries where there is a draw.
The performance of sbs is high for simple queries (e.g., 0.97 for row
ids) and decreases with query difficulty, from 0.44 for medium to 0.59
for hard queries. The columns’ performance follows a similar pattern,
highlighting that medium queries achieve a perfect match of 0.72. Across
all queries, the direct query generation method using Claude Opus 4.6
outperformed or performed similarly to the Gemini 2.5 Pro direct query
generation method. In addition, when comparing the results of the
step-by-step method, Claude Opus 4.6 outperformed or performed similarly
to Gemini 2.5 Pro for simple and hard queries.

Figure
[7](#S4.F7 "Figure 7 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the performance of the top six LLMs using the direct method, in
terms of the percentage of PM queries for row and column identifiers, as
a function of the degree of difficulty of the query. Likewise, Fig.
[8](#S4.F8 "Figure 8 ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the performance of the top six LLMs using the step-by-step method.
Figs.
[12](#A5.F12 "Figure 12 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
[13](#A5.F13 "Figure 13 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
and
[14](#A5.F14 "Figure 14 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
in Appendix
[E](#A5 "Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
show the progression over time of the LLM performance according to each
provider for the models evaluated (Claude, Gemini and GPT) for our T2S
task. The results of using GPT-5.2-Codex are shown in Fig.
[15](#A5.F15 "Figure 15 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").

![Refer to caption](2606.18108v1/results_dir_vs_step_v5_gemini25pro.png)

Figure 6: PM performance comparison of direct vs step-by-step with
self-correction using Gemini 2.5 Pro.

![Refer to caption](2606.18108v1/results_llm_comparison_v6_direct.png)

Figure 7: Performance comparison of the top six LLMs when using the
direct method, as a function of the level of query difficulty.

![Refer to caption](2606.18108v1/results_llm_comparison_v6.png)

Figure 8: Performance comparison of the top six LLMs when using the
step-by-step method, as a function of the level of query difficulty.

### 4.1 Analysis of errors

Three main kinds of execution errors can be distinguished. A timeout
error occurs when a query exceeds ALeRCE’s 2-minute execution limit,
e.g., when the model omits a specific condition needed to limit the
volume of the data requested. Alternatively, the model may misinterpret
the request, producing an incorrect condition, or place a condition in
the wrong scope (e.g., inside a subquery when it should be applied at
the outer level). The second category includes undefined objects,
tables, relations, columns, and so on –e.g., names of tables or columns
that are not in the database. The third category includes syntax and
other errors, such as: datatype mismatch (e.g., UNION types and double
precision cannot be matched); cardinality violation (e.g., more than one
row returned by a subquery used as an expression); and ambiguous column
(e.g., selecting the same column name from two tables without specifying
the table reference).

An error that we found during our system’s development was the LLMs’
tendency to rename columns. For example, in Table
[4](#S4.T4 "Table 4 ‣ 4.1 Analysis of errors ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
the golden column names ‘dist’ and ‘oid_catalog’ were changed to the
predicted columns ‘distance’ and ‘allwise_oid_catalog’, respectively.
The first error is semantically and logically correct, but it does not
match the real name. We solved this problem by parsing the queries after
execution using the Python library “sqlparse”.

|  Predicted SQL-Query |  [⬇](data:text/plain;base64,IFNFTEVDVAogICAgb2JqZWN0Lm9pZCBBUyB6dGZfb2lkLAogICAgYWxsd2lzZS5vaWRfY2F0YWxvZyBBUyBhbGx3aXNlX29pZF9jYXRhbG9nLAogICAgeG1hdGNoLmRpc3QgQVMgZGlzdGFuY2UgLi4u)  SELECT   object.oid AS ztf_oid,   allwise.oid_catalog AS allwise_oid_catalog,   xmatch.dist AS distance ... |
|----|----|
|  Extracted aliased columns |    ‘ztf_oid’, ‘allwise_oid_catalog’, ‘distance’, … |
|  Extracted real columns |    ‘oid’, ‘oid_catalog’, ‘dist’, … |
|  Gold columns |    ‘oid’, ‘oid_catalog’, ‘dist’, … |

Table 4: Example of parsing columns.

Figure
[9](#S4.F9 "Figure 9 ‣ 4.1 Analysis of errors ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
compares the execution errors obtained w/ and w/o self-correction when
using Claude Opus 4.6. The vast majority of errors are due to timeout.
Without self-correction, there are errors associated with undefined
columns and tables, cardinality violation, and invalid column
references. In Figure
[9](#S4.F9 "Figure 9 ‣ 4.1 Analysis of errors ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
all non-timeout errors are corrected by self-correction, except one
undefined column.

In addition to execution errors, there are semantic errors: the query is
executable but does not, or does not always, produce the intended
results. Semantic errors encompass a broad range of issues, among them
logical errors (e.g., confusing similarly named columns), language
comprehension, ambiguity issues and cases where the model lacks
specialized domain knowledge. These kinds of errors are not corrected by
self-correction. Fig.
[10](#S4.F10 "Figure 10 ‣ 4.1 Analysis of errors ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows that when using the step-by-step method with Claude Opus 4.6,
semantic errors are generally reduced or kept similar to those of the
direct method. It can be observed that for medium queries the percentage
of semantic errors is much higher for rows than for columns. This
discrepancy appears because column evaluation is much simpler than row
evaluation. Row precision and recall require the model to satisfy
multiple conditions simultaneously, including correctly identifying join
paths, applying filters, and constructing appropriate WHERE clauses.
Column selection, on the other hand, can be viewed as a classification
or association task; the LLM must identify which attributes are relevant
to the query and ensure that the generated SQL executes successfully
against the database schema. At the medium level, models can still
perform the column association reliably but begin to struggle with the
more compositional reasoning that row-level accuracy demands. The
analysis of errors for GPT-5.2-Codex and Gemini 3 Flash are shown in
Figs.
[16](#A5.F16 "Figure 16 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
and
[17](#A5.F17 "Figure 17 ‣ Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
in Appendix
[E](#A5 "Appendix E Additional results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").

![Refer to caption](2606.18108v1/ex_errors_claude46opus.png)

Figure 9: Number and type of execution errors using Claude Opus 4.6, a)
without self-correction and b) with self-correction, for the direct and
step-by-step query generation methods.

![Refer to
caption](2606.18108v1/percentage_error_difficulty_claude46opus.png)

Figure 10: Percentage of perfect queries, semantic errors, and execution
errors for the direct and step-by-step query generation methods using
Claude Opus 4.6.

### 4.2 Few-shot prompting

A few-shot prompt for text-to-SQL is a prompting strategy where an LLM
is given a small number of example NL/SQL pairs before asking it to
generate SQL for a new question. We tested Claude Opus 4.6 and Gemini
2.5 Pro with few-shot prompting. There are three main strategies for
example selection in the literature: random, it randomly samples k
examples from the available candidates; question similarity selection
([Liu et al., 2022](#bib.bib44), QTS;), it chooses k examples with the
most similar questions; masked question similarity selection ([Guo et
al., 2023](#bib.bib45), MQS;), it replaces table names, column names,
and values in all questions with a mask token, and computes the
similarities of the embeddings with the k-Nearest Neighbor (kNN)
algorithm. We made a few changes in the QTS and MQS strategies, in order
to adapt them to our dataset. For both strategies we use a similarity
search through embeddings, cosine similarity and top-k selection. For
MQS we mask the following elements: ZTF oids, numerical values, month
names, and ALeRCE classes. Table
[5](#S4.T5 "Table 5 ‣ 4.2 Few-shot prompting ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the results of 1-shot and 3-shot prompting when using Gemini 2.5
Pro (direct). The best results are obtained with the MQS strategy using
3-shot, which achieves a statistically significant increase for all
types of queries, as well as row/column ids, with respect to the 0-shot.
Notably the row (column) id performance increased 18 (14) points. QTS
3-shot also achieves statistically significant differences with 0-shot,
except for a draw in hard rows. Tables
[11](#A6.T11 "Table 11 ‣ Appendix F Few-shot prompting ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
and
[12](#A6.T12 "Table 12 ‣ Appendix F Few-shot prompting ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
in Appendix
[F](#A6 "Appendix F Few-shot prompting ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
show the results of few-shot prompting for the Gemini 2.5 Pro (sbs) and
Claude Opus 4.6 (sbs), respectively. Listing
[F.1](#LST1b "Listing F.1 ‣ Appendix F Few-shot prompting ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows an example of few-shot prompt.

|  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Few-shot | Selection | Gemini 2.5 Pro |  |  |  |  |  |
|  |  | Simple |  | Medium |  | Hard |  |
|  |  | Rows | Columns | Rows | Columns | Rows | Columns |
| 0-shot | \- | 0.86 $`\pm`$ 0.022 | 0.80 $`\pm`$ 0.016 | 0.42 $`\pm`$ 0.079 | 0.72 $`\pm`$ 0.063 | 0.36 $`\pm`$ 0.052 | 0.30 $`\pm`$ 0.000 |
| 1-shot | Random | 0.86 $`\pm`$ 0.017 $`\sim`$ | 0.82 $`\pm`$ 0.026 $`\sim`$ | 0.50 $`\pm`$ 0.000 $`\uparrow`$ | 0.84 $`\pm`$ 0.055 $`\uparrow`$ | 0.34 $`\pm`$ 0.055 $`\sim`$ | 0.34 $`\pm`$ 0.114 $`\sim`$ |
|  | QTS | 0.90 $`\pm`$ 0.015 $`\uparrow`$ | 0.85 $`\pm`$ 0.023 $`\uparrow`$ | 0.42 $`\pm`$ 0.042 $`\sim`$ | 0.79 $`\pm`$ 0.057 $`\uparrow`$ | 0.42 $`\pm`$ 0.063 $`\sim`$ | 0.27 $`\pm`$ 0.048 $`\sim`$ |
|  | MQS | 0.94 $`\pm`$ 0.022 $`\uparrow`$ | 0.89 $`\pm`$ 0.028 $`\uparrow`$ | 0.38 $`\pm`$ 0.045 $`\sim`$ | 0.76 $`\pm`$ 0.055 $`\sim`$ | 0.58 $`\pm`$ 0.045 $`\uparrow`$ | 0.36 $`\pm`$ 0.055 $`\sim`$ |
| 3-shot | Random | 0.92 $`\pm`$ 0.017 $`\uparrow`$ | 0.87 $`\pm`$ 0.014 $`\uparrow`$ | 0.44 $`\pm`$ 0.055 $`\sim`$ | 0.80 $`\pm`$ 0.071 $`\uparrow`$ | 0.34 $`\pm`$ 0.055 $`\sim`$ | 0.38 $`\pm`$ 0.045 $`\uparrow`$ |
|  | QTS | 0.97 $`\pm`$ 0.000 $`\uparrow`$ | 0.93 $`\pm`$ 0.017 $`\uparrow`$ | 0.58 $`\pm`$ 0.045 $`\uparrow`$ | 0.80 $`\pm`$ 0.000 $`\uparrow`$ | 0.40 $`\pm`$ 0.100 $`\sim`$ | 0.40 $`\pm`$ 0.000 $`\uparrow`$ |
|  | MQS | 0.94 $`\pm`$ 0.000 $`\uparrow`$ | 0.86 $`\pm`$ 0.017 $`\uparrow`$ | 0.60 $`\pm`$ 0.000 $`\uparrow`$ | 0.86 $`\pm`$ 0.055 $`\uparrow`$ | 0.48 $`\pm`$ 0.045 $`\uparrow`$ | 0.40 $`\pm`$ 0.000 $`\uparrow`$ |

Table 5: Few-shot PM results by query difficulty when using Gemini 2.5
Pro (direct).¹⁶¹⁶ 16 Notes. The best value per table column is bolded.
Arrows compare the few-shot setting against the 0-shot baseline for each
query difficulty using two-sided permutation tests on perfect-match
scores grouped by run; $`\uparrow`$/$`\downarrow`$ indicate significant
higher/lower results at $`\alpha=0.05`$ with no multiple-testing
correction, and $`\sim`$ indicates not significant.

## 5 Discussion

In this section we discuss two possible ways to improve our proposed
framework, especially from the viewpoint of putting in production our
T2S system in the near future: structured outputs and function/tool
calling. Regarding specialized tools, function calling can considerably
expand the model’s capabilities. In our NL/SQL dataset, the external
knowledge associated with some queries, which is mainly composed of MJD
dates and celestial coordinates (e.g., RA/Dec), was added by hand.
However, this kind of information can be obtained automatically through
function/tool calling. These tools leverage the model’s ability to
interpret the user’s request and format the appropriate input for each
function. Listing
[G.1](#LST1c "Listing G.1 ‣ Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows an example of a tool that converts dates found in the user request
to MJD format. Table
[13](#A7.T13 "Table 13 ‣ Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
(see Appendix
[G](#A7 "Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
) shows the PM results when using Claude Opus 4.6, Gemini 2.5 Pro, and
Gemini 3 Flash w/ and w/o the function calling for MJD. It can be
observed that the performance increases or remains the same for the sbs
query generation method, except for hard rows with Claude Opus 4.6. The
main point here is that the function/tool calling simplifies the
operation from the user viewpoint, and therefore it is highly
recommended to include it in production.

Using SQL as a native structured output means the LLM generates
executable database queries in a formally constrained syntax. This is
structured output because SQL grammar is formal, tables/columns have
schema constraints, and syntax validity is useful. For text-to-SQL
tasks, Claude, Gemini, and GPT models all support structured output, but
they implement it differently at the API and decoding levels. To obtain
structured outputs, we used the integrated structured outputs feature of
the API providers. OpenAI¹⁷¹⁷ 17
https://developers.openai.com/api/docs/guides/structured-outputs,
Anthropic¹⁸¹⁸ 18
https://platform.claude.com/docs/en/build-with-claude/structured-outputs,
and Google¹⁹¹⁹ 19
https://ai.google.dev/gemini-api/docs/structured-output have a specific
variable named “text_format”, “output_format”, and “response_format”,
respectively, where a JSON schema structure can be passed. Another
option is to define a Pydantic BaseModel structure, as shown in all the
model guide pages, and selecting the “ $`{\rm extra}=`forbid^{\prime}`$
” configuration to enable strict mode to comply with strict JSON schema
requirements and matching. Some benefits of structured outputs include
reliable type-safety, seamless system integration, reduced
hallucinations, and simpler prompting to obtain consistent formatting.
We tested a type of structured output for the decomposition stage of the
step-by-step method for medium/hard queries, defined in Pydantic as
shown in Listing
[G.2](#LST2b "Listing G.2 ‣ Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
(see Appendix
[G](#A7 "Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"))
and that can be transformed to a JSON schema. Listing
[G.3](#LST3a "Listing G.3 ‣ Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the Pydantic structure transformed into a JSON dictionary schema
that is the format given to the model, forcing the latter to return a
list of steps to build the required SQL, and then joining all these
steps into one large query in the SQL generation stage. We tested this
format with the medium and hard queries. Table
[14](#A7.T14 "Table 14 ‣ Appendix G Function calling and structured outputs ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system")
shows the performance with the normal output and the structured
step-by-step output for Gemini 3 Flash, Gemini 2.5 Pro, and Claude Opus
4.6. The models did not show a significant performance improvement.
Gemini 3 Flash obtained better results for columns for medium queries,
while for rows in medium queries, a degraded performance is obtained for
Gemini 3 Flash and Claude Opus 4.6. Nevertheless, some cases showed a
more consistent output with more successful runs, reducing execution
errors. This approach is worth to be explored in more detail in future
work.

A natural extension of the current self-correction module would be to
constrain its output via a typed schema (e.g., a Pydantic-validated
correction object carrying identifier-replacement records restricted to
the database vocabulary, applied deterministically on the SQL Abstract
Syntax Tree (AST)), with a free-form fallback for syntax, type, and
structural errors. This would prevent hallucinated table and column
names in the most common error class while preserving the flexibility
required for user-requested aliases and complex query structures. This
is a valuable direction for future work and we plan to explore it,
particularly the post-generation validation approach, which is the most
compatible with our current API-based pipeline.

## 6 Conclusions

In this work, we proposed a framework for performing the text-to-SQL
parsing task in astronomy using in-context learning with LLMs. With this
aim, we built a dataset of NL/SQL queries for the ALeRCE database. We
show that the proposed framework outperforms the direct inference method
and that self-correction, in general, improves results. In the
performance comparison of thirteen LLMs, we found that Claude Opus 4.6,
Gemini 2.5 Pro, Gemini 3 Flash and GPT 5.2 Codex are the best LLMs for
the task at hand. In future work, we plan to explore technologies beyond
classic prompt engineering, such as Retrieval-Augmented Generation (RAG)
and fine-tuning LLMs on generating SQL outputs. Future work will
evaluate the proposed framework against both specialized text-to-SQL
systems, e.g., SQLCoder²⁰²⁰ 20 https://github.com/defog-ai/sqlcoder, and
modern open-weight reasoning models such as DeepSeek, Qwen, and LLaMA
variants. In addition, developing a more comprehensive self-review
process that evaluates semantic correctness, not only execution errors,
is a promising direction for future work. Our ultimate goal is to
implement an online virtual assistant for ALeRCE users in the Rubin era.

###### Acknowledgements.

This work gratefully acknowledges funding from ANID: Millennium Science
Initiative, AIM23-0001 (P.A.E, J.E-M, F.F., G.C.-V. A.M.M.A, G.P.,
F.E.B., M.C., R.D.); ANID/Basal (CATA) grant FB210003 (F.F., M.C.,
F.E.B.); and FONDECYT Regular grants 1220829 (P.A.E.), 1231877 (G.C.V),
1241005 (F.E.B.), and 1231637 (M.C.). This work was supported by Centro
de Modelamiento Matemático (CMM) BASAL fund FB210005 from ANID-Chile.
A.B acknowledges support from the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) under Germany’s Excellence Strategy – EXC
2094 – 390783311. We acknowledge Zhen Guo and Arti Joshi for their
contribution in providing samples for the dataset used in this paper.

## References

- Alexander et al. (2026) K. D. Alexander, R. Margutti, S. Gomez, M.
  Stroh, R. Chornock, T. Laskar, Y. Cendes, E. Berger, T. Eftekhari, N.
  Franz, et al. The multiwavelength context of delayed radio emission in
  tidal disruption events: evidence for accretion-driven outflows. ApJ
  1000 (1), pp. 139. External Links:
  [Link](https://doi.org/10.3847/1538-4357/ae40ab),
  [Document](https://dx.doi.org/10.3847/1538-4357/ae40ab) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Arévalo et al. (2024) P. Arévalo, E. López-Navas, M.
  Martínez-Aldama, P. Lira, S. Bernal, P. Sánchez-Sáez, M. Salvato, L.
  Hernández-García, C. Ricci, A. Merloni, et al. A newborn active
  galactic nucleus in a star-forming galaxy. A&A 683, pp. L8. Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Bellm et al. (2019) E. C. Bellm, S. R. Kulkarni, M. J. Graham, R.
  Dekany, R. M. Smith, R. Riddle, F. J. Masci, G. Helou, T. A.
  Prince, S. M. Adams, et al. The zwicky transient facility: system
  overview, performance, and first results. PASP 131 (995), pp. 018002.
  External Links: [Link](https://doi.org/10.1088/1538-3873/aaecbe),
  [Document](https://dx.doi.org/10.1088/1538-3873/aaecbe) Cited by:
  [§1](#S1.p6.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Carrasco-Davis et al. (2021) R. Carrasco-Davis, E. Reyes, C.
  Valenzuela, F. Förster, P. A. Estévez, G. Pignata, F. E. Bauer, I.
  Reyes, P. Sánchez-Sáez, G. Cabrera-Vives, et al. Alert classification
  for the alerce broker system: the real-time stamp classifier. AJ 162
  (6), pp. 231. Cited by:
  [§1](#S1.p6.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Chang and Fosler-Lussier (2023) S. Chang and E. Fosler-Lussier How to
  prompt LLMs for text-to-SQL: a study in zero-shot, single-domain, and
  cross-domain settings. In NeurIPS 2023 Second Table Representation
  Learning Workshop, External Links:
  [Link](https://openreview.net/forum?id=5sOZNkkKh3) Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Ciuca et al. (2023) I. Ciuca, Y. Ting, S. Kruk, and K. Iyer Harnessing
  the power of adversarial prompting and large language models for
  robust hypothesis generation in astronomy. ArXiv e-prints. External
  Links: 2306.11648, [Link](https://arxiv.org/abs/2306.11648) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§1](#S1.p2.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Coughlin et al. (2023) M. W. Coughlin, J. S. Bloom, G. Nir, S.
  Antier, T. J. du Laz, S. f. van der Walt, A. Crellin-Quick, T.
  Culino, D. A. Duev, D. A. Goldstein, B. F. Healy, V. Karambelkar, J.
  Lilleboe, K. M. Shin, L. P. Singer, T. Ahumada, S. Anand, E. C.
  Bellm, R. Dekany, M. J. Graham, M. M. Kasliwal, I. Kostadinova, R. W.
  Kiendrebeogo, S. R. Kulkarni, S. Jenkins, N. LeBaron, A. A.
  Mahabal, J. D. Neill, B. Parazin, J. Peloton, D. A. Perley, R.
  Riddle, B. Rusholme, J. van Santen, J. Sollerman, R. Stein, D.
  Turpin, A. Wold, C. Amat, A. Bonnefon, A. Bonnefoy, M. Flament, F.
  Kerkow, S. Kishore, S. Jani, S. K. Mahanty, C. Liu, L. Llinares, J.
  Makarison, A. Olliéric, I. Perez, L. Pont, and V. Sharma A data
  science platform to enable time-domain astronomy. ApJS 267 (2),
  pp. 31. External Links:
  [Document](https://dx.doi.org/10.3847/1538-4365/acdee1),
  [Link](https://doi.org/10.3847%2F1538-4365%2Facdee1) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§1](#S1.p3.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Dong et al. (2023) X. Dong, C. Zhang, Y. Ge, Y. Mao, Y. Gao, J.
  Lin, D. Lou, et al. C3: zero-shot text-to-sql with chatgpt. ArXiv
  e-prints. External Links: 2307.07306 Cited by:
  [§2.2](#S2.SS2.p4.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Förster et al. (2021) F. Förster, G. Cabrera-Vives, E.
  Castillo-Navarrete, P. Estévez, P. Sánchez-Sáez, J. Arredondo, F.
  Bauer, R. Carrasco-Davis, M. Catelan, F. Elorrieta, et al. The
  automatic learning for the rapid classification of events (alerce)
  alert broker. AJ 161 (5), pp. 242. Cited by:
  [§1](#S1.p6.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Gao et al. (2024) D. Gao, H. Wang, Y. Li, X. Sun, Y. Qian, B. Ding,
  and J. Zhou Text-to-sql empowered by large language models: a
  benchmark evaluation. Proc. VLDB Endow. 17 (5), pp. 1132. External
  Links: ISSN 2150-8097,
  [Link](https://doi.org/10.14778/3641204.3641221),
  [Document](https://dx.doi.org/10.14778/3641204.3641221) Cited by:
  [§2.2](#S2.SS2.p4.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§3.5](#S3.SS5.p1.1 "3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Ghosh et al. (2022) M. Ghosh, P. Santra, S. A. Iqbal, and P.
  Basuchowdhuri Astro-mt5: entity extraction from astrophysics
  literature using mt5 language model. In Proceedings of the first
  Workshop on Information Extraction from Scientific Publications, T.
  Ghosal, S. Blanco-Cuaresma, A. Accomazzi, R. M. Patton, F. Grezes,
  and T. Allen (Eds.), pp. 100. External Links:
  [Link](https://aclanthology.org/2022.wiesp-1.12/),
  [Document](https://dx.doi.org/10.18653/v1/2022.wiesp-1.12) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Gomez and Gezari (2023) S. Gomez and S. Gezari The search for
  thermonuclear transients from the tidal disruption of a white dwarf by
  an intermediate-mass black hole. ApJ 955 (1), pp. 46. External Links:
  [Link](https://doi.org/10.3847/1538-4357/acefbc),
  [Document](https://dx.doi.org/10.3847/1538-4357/acefbc) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Grezes et al. (2024) F. Grezes, S. Blanco-Cuaresma, A.
  Accomazzi, M. J. Kurtz, G. Shapurian, E. Henneken, C. S. Grant, D. M.
  Thompson, R. Chyla, S. McDonald, et al. Building astrobert, a language
  model for astronomy & astrophysics. Astronomical Data Analysis
  Software and Systems XXXI 535, pp. 119. Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Guo et al. (2023) C. Guo, Z. Tian, J. Tang, P. Wang, Z. Wen, K. Yang,
  and T. Wang Prompting gpt-3.5 for text-to-sql with de-semanticization
  and skeleton retrieval. In 20th Pacific Rim International Conference
  on Artificial Intelligence, PRICAI, Proceedings, Part II, pp. 262.
  External Links: ISBN 978-981-99-7021-6,
  [Link](https://doi.org/10.1007/978-981-99-7022-3_23),
  [Document](https://dx.doi.org/10.1007/978-981-99-7022-3%5F23) Cited
  by:
  [§4.2](#S4.SS2.p1.1 "4.2 Few-shot prompting ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Ivezić et al. (2019) Ž. Ivezić, S. M. Kahn, J. A. Tyson, B. Abel, E.
  Acosta, R. Allsman, D. Alonso, Y. AlSayyad, S. F. Anderson, J. Andrew,
  et al. LSST: from science drivers to reference design and anticipated
  data products. ApJ 873 (2), pp. 111. External Links:
  [Link](https://doi.org/10.3847/1538-4357/ab042c),
  [Document](https://dx.doi.org/10.3847/1538-4357/ab042c) Cited by:
  [§1](#S1.p6.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Iyer et al. (2024) K. G. Iyer, M. Yunus, C. O’Neill, C. Ye, A. Hyk, K.
  McCormick, I. Ciucă, J. F. Wu, A. Accomazzi, S. Astarita, R.
  Chakrabarty, J. Cranney, A. Field, T. Ghosal, M. Ginolfi, M.
  Huertas-Company, M. Jabłońska, S. Kruk, H. Liu, G. Marchidan, R.
  Mistry, J. P. Naiman, J. E. G. Peek, M. Polimera, S. J. Rodríguez
  Méndez, K. Schawinski, S. Sharma, M. J. Smith, Y. Ting, M. Walmsley,
  and (UniverseTBD) Pathfinder: a semantic framework for literature
  review and knowledge discovery in astronomy. ApJS 275 (2), pp. 38.
  External Links: [Link](https://doi.org/10.3847/1538-4365/ad7c43),
  [Document](https://dx.doi.org/10.3847/1538-4365/ad7c43) Cited by:
  [§1](#S1.p2.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Joseph et al. (2026) S. Joseph, S. M. Husain, S. Offner, S. Juneau, P.
  Torrey, A. Bolton, J. Farias, N. Gaffney, G. Durrett, and J. J. Li
  Astrovisbench: a code benchmark for scientific computing and
  visualization in astronomy. In The Thirty-ninth Annual Conference on
  Neural Information Processing Systems Datasets and Benchmarks Track,
  Vol. 38. External Links:
  [Link](https://openreview.net/forum?id=qXiTFAgEx4) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§1](#S1.p4.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Katsogiannis-Meimarakis and Koutrika (2023) G. Katsogiannis-Meimarakis
  and G. Koutrika A survey on deep learning approaches for text-to-sql.
  The VLDB Journal 32 (4), pp. 905. External Links: ISSN 0949-877X,
  [Link](https://doi.org/10.1007/s00778-022-00776-8),
  [Document](https://dx.doi.org/10.1007/s00778-022-00776-8) Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Khot et al. (2023) T. Khot, H. Trivedi, M. Finlayson, Y. Fu, K.
  Richardson, P. Clark, and A. Sabharwal Decomposed prompting: a modular
  approach for solving complex tasks. In The Eleventh International
  Conference on Learning Representations, External Links:
  [Link](https://openreview.net/forum?id=_nGgzQjzaRy) Cited by: [item
  3](#S3.I2.i3.p1.1 "In 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Kojima et al. (2022) T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y.
  Iwasawa Large language models are zero-shot reasoners. In Proceedings
  of the 36th International Conference on Neural Information Processing
  Systems, Vol. 35, pp. 22199. External Links: ISBN 9781713871088 Cited
  by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Li et al. (2025) J. Li, G. Li, Y. Li, and Z. Jin Structured
  chain-of-thought prompting for code generation. ACM Trans. Softw. Eng.
  Methodol. 34 (2), pp. 1. External Links: ISSN 1049-331X,
  [Link](https://doi.org/10.1145/3690635),
  [Document](https://dx.doi.org/10.1145/3690635) Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Li et al. (2024) J. Li, B. Hui, G. Qu, J. Yang, B. Li, B. Li, B.
  Wang, B. Qin, R. Geng, N. Huo, X. Zhou, C. Ma, G. Li, K. C.C.
  Chang, F. Huang, R. Cheng, and Y. Li Can LLM already serve as a
  database interface? a BIg bench for large-scale database grounded
  text-to-SQLs. Advances in Neural Information Processing Systems 36.
  Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§2.1](#S2.SS1.p2.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§3.6](#S3.SS6.p14.1 "3.6 Prompt engineering ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§3.7](#S3.SS7.p1.1 "3.7 Evaluation metrics ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Liu et al. (2023) H. Liu, Z. Teng, L. Cui, C. Zhang, Q. Zhou, and Y.
  Zhang LogiCoT: logical chain-of-thought instruction tuning. In
  Findings of the Association for Computational Linguistics: EMNLP 2023,
  pp. 2908. External Links:
  [Link](https://openreview.net/forum?id=qlCtkvgQJH),
  [Document](https://dx.doi.org/10.18653/v1/2023.findings-emnlp.191)
  Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Liu et al. (2022) J. Liu, D. Shen, Y. Zhang, W. B. Dolan, L. Carin,
  and W. Chen What makes good in-context examples for gpt-3?. In
  Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd
  workshop on knowledge extraction and integration for deep learning
  architectures, pp. 100. External Links:
  [Link](https://aclanthology.org/2022.deelio-1.10/),
  [Document](https://dx.doi.org/10.18653/v1/2022.deelio-1.10) Cited by:
  [§4.2](#S4.SS2.p1.1 "4.2 Few-shot prompting ‣ 4 Results ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- López-Navas et al. (2022) E. López-Navas, M. Martínez-Aldama, S.
  Bernal, P. Sánchez-Sáez, P. Arévalo, M. J. Graham, L.
  Hernández-García, P. Lira, and P. Rojas Lobos Confirming new
  changing-look agns discovered through optical variability using a
  random forest-based light-curve classifier. MNRAS 513 (1), pp. L57.
  Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Magee et al. (2023) M. Magee, A. Sainz de Murieta, T. Collett, and W.
  Enzi A search for gravitationally lensed supernovae within the zwicky
  transient facility public survey. MNRAS 525 (1), pp. 542. External
  Links: ISSN 0035-8711, [Link](https://doi.org/10.1093/mnras/stad2263),
  [Document](https://dx.doi.org/10.1093/mnras/stad2263) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Müller-Bravo et al. (2025) T. E. Müller-Bravo, L. Galbany, M.
  Stritzinger, C. Ashall, E. Baron, C. R. Burns, P. Höflich, N.
  Morrell, M. Phillips, N. B. Suntzeff, et al. Analyzing type ia
  supernovae near-infrared light curves with principal component
  analysis. A&A 702, pp. A134. External Links:
  [Link](https://doi.org/10.1051/0004-6361/202555078),
  [Document](https://dx.doi.org/10.1051/0004-6361/202555078) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Nguyen et al. (2023) T. D. Nguyen, Y. Ting, I. Ciuca, C. O’Neill, Z.
  Sun, M. Jabłońska, S. Kruk, E. Perkowski, J. Miller, J. J. J. Li, J.
  Peek, K. Iyer, T. Rozanski, P. Khetarpal, S. Zaman, D. Brodrick, S. J.
  Rodriguez Mendez, T. Bui, A. Goodman, A. Accomazzi, J. Naiman, J.
  Cranney, K. Schawinski, and R. Raileanu Astrollama: towards
  specialized foundation models in astronomy. In Proceedings of the
  Second Workshop on Information Extraction from Scientific
  Publications, pp. 49. External Links:
  [Link](https://aclanthology.org/2023.wiesp-1.7/),
  [Document](https://dx.doi.org/10.18653/v1/2023.wiesp-1.7) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Oshikiri et al. (2024) K. Oshikiri, M. Tanaka, N. Tominaga, T.
  Morokuma, I. Takahashi, Y. Tampo, H. Hamidani, N. Arima, K.
  Arimatsu, T. Kasuga, et al. A search for extragalactic fast optical
  transients in the tomo-e gozen high-cadence survey. MNRAS 527 (1),
  pp. 334. External Links: ISSN 0035-8711,
  [Link](https://doi.org/10.1093/mnras/stad3184),
  [Document](https://dx.doi.org/10.1093/mnras/stad3184) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Perkowski et al. (2024) E. Perkowski, R. Pan, T. D. Nguyen, Y.
  Ting, S. Kruk, T. Zhang, C. O’Neill, M. Jablonska, Z. Sun, M. J.
  Smith, H. Liu, K. Schawinski, K. Iyer, I. Ciucă, and UniverseTBD
  AstroLLaMA-chat: scaling astrollama with conversational and diverse
  datasets. Research Notes of the AAS 8 (1), pp. 7. External Links:
  [Link](https://doi.org/10.3847/2515-5172/ad1abe),
  [Document](https://dx.doi.org/10.3847/2515-5172/ad1abe) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§1](#S1.p2.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Pomeroy and Norris (2024) R. T. Pomeroy and M. A. Norris A search for
  intermediate-mass black holes in compact stellar systems through
  optical emissions from tidal disruption events. MNRAS 530 (3),
  pp. 3043. External Links: ISSN 0035-8711,
  [Link](https://doi.org/10.1093/mnras/stae960),
  [Document](https://dx.doi.org/10.1093/mnras/stae960) Cited by:
  [§3.1](#S3.SS1.p2.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Pourreza and Rafiei (2023) M. Pourreza and D. Rafiei DIN-sql:
  decomposed in-context learning of text-to-sql with self-correction. In
  Proceedings of the 37th International Conference on Neural Information
  Processing Systems, Vol. 36, pp. 36339. External Links:
  [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/72223cc66f63ca1aa59edaec1b3670e6-Paper-Conference.pdf)
  Cited by:
  [§2.2](#S2.SS2.p3.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§3.5](#S3.SS5.p1.1 "3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Qin et al. (2022) B. Qin, B. Hui, L. Wang, M. Yang, J. Li, B. Li, R.
  Geng, R. Cao, J. Sun, L. Si, et al. A survey on text-to-sql parsing:
  concepts, methods, and future directions. ArXiv e-prints. External
  Links: 2208.13629 Cited by:
  [§2.1](#S2.SS1.p1.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§2.1](#S2.SS1.p2.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Sahoo et al. (2024) P. Sahoo, A. K. Singh, S. Saha, V. Jain, S.
  Mondal, and A. Chadha A systematic survey of prompt engineering in
  large language models: techniques and applications. ArXiv e-prints.
  External Links: 2402.07927, [Link](https://arxiv.org/abs/2402.07927)
  Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Sánchez-Sáez et al. (2021) P. Sánchez-Sáez, I. Reyes, C.
  Valenzuela, F. Förster, S. Eyheramendy, F. Elorrieta, F. Bauer, G.
  Cabrera-Vives, P. Estévez, M. Catelan, et al. Alert classification for
  the alerce broker system: the light curve classifier. AJ 161 (3),
  pp. 141. External Links:
  [Link](https://doi.org/10.3847/1538-3881/abd5c1),
  [Document](https://dx.doi.org/10.3847/1538-3881/abd5c1) Cited by:
  [§1](#S1.p6.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§3.1](#S3.SS1.p1.1 "3.1 ALeRCE database ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Shao et al. (2024) W. Shao, R. Zhang, P. Ji, D. Fan, Y. Hu, X. Yan, C.
  Cui, Y. Tao, L. Mi, and L. Chen Astronomical knowledge entity
  extraction in astrophysics journal articles via large language models.
  Research in Astronomy and Astrophysics 24 (6), pp. 065012. External
  Links: [Document](https://dx.doi.org/10.1088/1674-4527/ad3d15),
  [Link](https://dx.doi.org/10.1088/1674-4527/ad3d15) Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Sotnikov and Chaikova (2023) V. Sotnikov and A. Chaikova Language
  models for multimessenger astronomy. Galaxies 11 (3), pp. 63. External
  Links: [Link](https://www.mdpi.com/2075-4434/11/3/63), ISSN 2075-4434
  Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Wang et al. (2025) B. Wang, C. Ren, J. Yang, X. Liang, J. Bai, L.
  Chai, Z. Yan, Q. Zhang, D. Yin, X. Sun, et al. Mac-sql: a multi-agent
  collaborative framework for text-to-sql. In Proceedings of the 31st
  International Conference on Computational Linguistics, pp. 540.
  External Links: [Link](https://aclanthology.org/2025.coling-main.36/)
  Cited by:
  [§2.2](#S2.SS2.p4.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Wei et al. (2022) J. Wei, X. Wang, D. Schuurmans, M. Bosma, b.
  ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou Chain-of-thought
  prompting elicits reasoning in large language models. In Advances in
  Neural Information Processing Systems, Vol. 35, pp. 24824. External
  Links:
  [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)
  Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Ye et al. (2025) C. Ye, S. Yuan, S. Cooray, S. Dillmann, I. L.
  Roque, D. Baron, P. Frank, S. Martin-Alvarez, N. Koblischke, F. J. Qu,
  et al. ReplicationBench: can ai agents replicate astrophysics research
  papers?. ArXiv e-prints. External Links: 2510.24591 Cited by:
  [§1](#S1.p1.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§1](#S1.p4.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Yu et al. (2018) T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z.
  Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang, and D. Radev Spider: a
  large-scale human-labeled dataset for complex and cross-domain
  semantic parsing and text-to-SQL task. In Proceedings of the 2018
  Conference on Empirical Methods in Natural Language Processing,
  pp. 3911. External Links: [Link](https://aclanthology.org/D18-1425/),
  [Document](https://dx.doi.org/10.18653/v1/D18-1425) Cited by:
  [§2.1](#S2.SS1.p2.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Zhang et al. (2023) Y. Zhang, J. Deriu, G. Katsogiannis-Meimarakis, C.
  Kosten, G. Koutrika, and K. Stockinger ScienceBenchmark: a complex
  real-world benchmark for evaluating natural language to sql systems.
  Proc. VLDB Endow. 17 (4), pp. 685. External Links: ISSN 2150-8097,
  [Link](https://doi.org/10.14778/3636218.3636225),
  [Document](https://dx.doi.org/10.14778/3636218.3636225) Cited by:
  [§1](#S1.p5.1 "1 Introduction ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system"),
  [§2.1](#S2.SS1.p2.1 "2.1 Background on text-to-SQL parsing ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Zhao et al. (2026) W. X. Zhao, K. Zhou, J. Li, T. Tang, Z. Dong, Y.
  Hou, B. Zhang, Y. Min, J. Zhang, P. Liu, X. Wang, Y. Du, C. Yang, Y.
  Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, Y. Hu, J.
  Nie, and J. Wen A Survey of Large Language Models. Frontiers of
  Computer Science 20 (12), pp. 2012627. External Links: ISSN 2095-2236,
  [Link](https://doi.org/10.1007/s11704-026-60308-3),
  [Document](https://dx.doi.org/10.1007/s11704-026-60308-3) Cited by:
  [§2.2](#S2.SS2.p1.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Zhao et al. (2024) X. Zhao, M. Li, W. Lu, C. Weber, J. Lee, K. Chu,
  and S. Wermter Enhancing zero-shot chain-of-thought reasoning in large
  language models through logic. In Proceedings of the 2024 Joint
  International Conference on Computational Linguistics, Language
  Resources and Evaluation (LREC-COLING 2024), pp. 6144. External Links:
  [Link](https://aclanthology.org/2024.lrec-main.543/) Cited by:
  [§2.2](#S2.SS2.p2.1 "2.2 Prompt engineering for the T2S task ‣ 2 Background ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").
- Zhou et al. (2023) D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X.
  Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. V. Le, and E. H. Chi
  Least-to-most prompting enables complex reasoning in large language
  models. In The Eleventh International Conference on Learning
  Representations, External Links:
  [Link](https://openreview.net/forum?id=WZH7099tgfM) Cited by: [item
  3](#S3.I2.i3.p1.1 "In 3.5 Proposed framework for the T2S task ‣ 3 Methodology ‣ Querying an astronomical database using large language models: the ALeRCE text-to-SQL system").

## Appendix A  ALeRCE database

![Refer to caption](2606.18108v1/db_nlp.png)

Figure 11: Entity-relationship diagram of the ALeRCE database.

## Appendix B Examples of queries

|  |  |
|----|----|
| Request |  For objects with ZTF identifiers ’ZTF21aaobkmg’, ’ZTF21aaomuka’, find all rows in the ’probability’ table and light curve classifier that have ranking 1 or 2. Return all columns from such table, sort by ranking |
|  Gold SQL Query |  [⬇](data:text/plain;base64,IFNFTEVDVAogICAgKgogRlJPTQogICAgcHJvYmFiaWxpdHkKIFdIRVJFCiAgICBvaWQgSU4gKCdaVEYyMWFhb2JrbWcnLCdaVEYyMWFhb211a2EnKSBBTkQKICAgIGNsYXNzaWZpZXJfbmFtZSA9ICdsY19jbGFzc2lmaWVyJyBBTkQKICAgIHJhbmtpbmcgPD0gMgogT1JERVIgQlkgcmFua2luZw==)  SELECT   \*   FROM   probability   WHERE   oid IN (’ZTF21aaobkmg’,’ZTF21aaomuka’) AND   classifier_name = ’lc_classifier’ AND   ranking \<= 2   ORDER BY ranking |

Table 6: Example of a simple query.

|  |  |
|----|----|
| Request |  For Solar System identifiers ’2003FP134’ and ’2009UK56’, get all detections for all ZTF objects that lie within 2 arcsec from any of them. Return the following columns, sort by MPC name and detection date: all columns from the ’ss_ztf’ table; and detection date, filter identifier, isdiffpos flag, RA Dec coordinates, and difference magnitude (and its uncertainty) |
|  Gold SQL Query |  [⬇](data:text/plain;base64,U0VMRUNUCiAgICBzc196dGYuKiwgZGV0ZWN0aW9uLm1qZCwgZGV0ZWN0aW9uLmZpZCwgZGV0ZWN0aW9uLmlzZGlmZnBvcywgZGV0ZWN0aW9uLnJhLCBkZXRlY3Rpb24uZGVjLCBkZXRlY3Rpb24ubWFncHNmLCBkZXRlY3Rpb24uc2lnbWFwc2YKRlJPTQogICAgc3NfenRmCklOTkVSIEpPSU4gZGV0ZWN0aW9uIE9OIHNzX3p0Zi5vaWQgPSBkZXRlY3Rpb24ub2lkIEFORCBzc196dGYuY2FuZGlkID0gZGV0ZWN0aW9uLmNhbmRpZApXSEVSRQogICAgc3NuYW1lbnIgSU4gKCcyMDAzRlAxMzQnLCcyMDA5VUs1NicpIEFORAogICAgc3NkaXN0bnIgPDIKT1JERVIgQlkgc3NuYW1lbnIsIG1qZA==) SELECT   ss_ztf.\*, detection.mjd, detection.fid, detection.isdiffpos, detection.ra, detection.dec, detection.magpsf, detection.sigmapsf  FROM   ss_ztf  INNER JOIN detection ON ss_ztf.oid = detection.oid AND ss_ztf.candid = detection.candid  WHERE   ssnamenr IN (’2003FP134’,’2009UK56’) AND   ssdistnr \<2  ORDER BY ssnamenr, mjd |

Table 7: Example of a medium query.

|  |  |
|----|----|
| Request |  Find all detections, non-detections and forced photometry points for the ZTF object ’ZTF24aamtvxb’. Return all epochs in the same output table, including the following columns: ZTF identifier, epoch date, filter identifier, isdiffpos flag, detection difference magnitude and its uncertainty, 5-sigma magnitude limit, forced difference magnitude and its uncertainty, and a column named ’table’ (that contains the name of the table of origin for each epoch) |
| Gold SQL Query |  [⬇](data:text/plain;base64,IFNFTEVDVAogICAgb2lkLCBtamQsIGZpZCwgaXNkaWZmcG9zLCBtYWdwc2YsIHNpZ21hcHNmLCBOVUxMIGFzIGRpZmZtYWdsaW0sIENBU1QoTlVMTCBBUyBiaWdpbnQpIGFzIG1hZywgQ0FTVChOVUxMIEFTIGJpZ2ludCkgYXMgZV9tYWcsICdkZXRlY3Rpb24nIGFzIHRhYmxlCkZST00KICAgIGRldGVjdGlvbgpXSEVSRQogICAgb2lkID0gJ1pURjI0YWFtdHZ4YicgVU5JT04gQUxMIFNFTEVDVCBvaWQsIG1qZCwgZmlkLCBOVUxMIGFzIGlzZGlmZnBvcywgTlVMTCBhcyBtYWdwc2YsIE5VTEwgYXMgc2lnbWFwc2YsIGRpZmZtYWdsaW0sIENBU1QoTlVMTCBBUyBiaWdpbnQpIGFzIG1hZywgQ0FTVChOVUxMIEFTIGJpZ2ludCkgYXMgZV9tYWcsICdub25fZGV0ZWN0aW9uJyBhcyB0YWJsZQpGUk9NCiAgICBub25fZGV0ZWN0aW9uIFdIRVJFIG9pZCA9ICdaVEYyNGFhbXR2eGInClVOSU9OIEFMTApTRUxFQ1QKICAgIG9pZCwgbWpkLCBmaWQsIGlzZGlmZnBvcywgTlVMTCBhcyBtYWdwc2YsIE5VTEwgYXMgc2lnbWFwc2YsIE5VTEwgYXMgZGlmZm1hZ2xpbSwgbWFnLCBlX21hZywgJ2ZvcmNlZF9waG90b21ldHJ5JyBhcyB0YWJsZQpGUk9NCiAgICBmb3JjZWRfcGhvdG9tZXRyeQpXSEVSRQogICAgb2lkID0gJ1pURjI0YWFtdHZ4Yic=)  SELECT   oid, mjd, fid, isdiffpos, magpsf, sigmapsf, NULL as diffmaglim, CAST(NULL AS bigint) as mag, CAST(NULL AS bigint) as e_mag, ’detection’ as table  FROM   detection  WHERE   oid = ’ZTF24aamtvxb’ UNION ALL SELECT oid, mjd, fid, NULL as isdiffpos, NULL as magpsf, NULL as sigmapsf, diffmaglim, CAST(NULL AS bigint) as mag, CAST(NULL AS bigint) as e_mag, ’non_detection’ as table  FROM   non_detection WHERE oid = ’ZTF24aamtvxb’  UNION ALL  SELECT   oid, mjd, fid, isdiffpos, NULL as magpsf, NULL as sigmapsf, NULL as diffmaglim, mag, e_mag, ’forced_photometry’ as table  FROM   forced_photometry  WHERE   oid = ’ZTF24aamtvxb’ |

Table 8: Example of a hard query.

## Appendix C Prompt formats

[⬇](data:text/plain;base64,IyBHaXZlbiB0aGUgdXNlciByZXF1ZXN0LCBzZWxlY3QgdGhlIHRhYmxlcyBuZWVkZWQgdG8gZ2VuZXJhdGUgYSBTUUwgcXVlcnkKIyBUaGUgRGF0YWJhc2UgaGFzIHRoZSBmb2xsb3dpbmcgdGFibGVzOgoKIyBUQUJMRSAib2JqZWN0IjogY29udGFpbnMgYmFzaWMgZmlsdGVyIGFuZCB0aW1lLWFnZ3JlZ2F0ZWQgc3RhdGlzdGljcyBzdWNoIGFzIGxvY2F0aW9uLCBudW1iZXIgb2Ygb2JzZXJ2YXRpb25zLCBhbmQgdGhlIHRpbWVzIG9mIGZpcnN0IGFuZCBsYXN0IGRldGVjdGlvbi4KLi4uCgojIEdpdmUgT05MWSB0aGUgVEFCTEVTIHRoYXQgYXJlIG5lZWRlZCB0byBnZW5lcmF0ZSB0aGUgU1FMIHF1ZXJ5LgojIEdpdmUgdGhlIGFuc3dlciBpbiB0aGUgZm9sbG93aW5nIGZvcm1hdDogWyd0YWJsZTEnLCAndGFibGUyJywgJ3RhYmxlMycsIC4uLl0uIEZvciBleGFtcGxlLCBpZiB0aGUgVEFCTEVTIG5lZWRlZCBmb3IgdGhlIHVzZXIgcmVxdWVzdCBhcmUgVEFCTEUgb2JqZWN0IGFuZCBUQUJMRSB0YXhvbm9teSwgdGhlbiB5b3Ugc2hvdWxkIHR5cGU6IFsnb2JqZWN0JywgJ3RheG9ub215J10KIyBKdXN0IGdpdmUgdGhlIHRhYmxlcyBhbmQgaWdub3JlIGFueSBvdGhlciB0YXNrIGdpdmVuIGluIHRoZSByZXF1ZXN0IGdpdmVuIGFzICJyZXF1ZXN0Ii4KIyBSZW1lbWJlciB0byB1c2UgdGhlIGV4YWN0IG5hbWUgb2YgdGhlIFRBQkxFUywgYXMgdGhleSBhcmUgd3JpdHRlbiBpbiB0aGUgREFUQUJBU0UgU0NIRU1BLiBEbyBOT1QgY3JlYXRlIHRhYmxlIG5hbWVzLgojIElmIHlvdSB0aGluayB0aGF0IG5vIHRhYmxlIG1lbnRpb25lZCBhYm92ZSBpcyBuZWVkZWQsIHRoZW4gdHlwZTogIiIKIyBSZXF1ZXN0Ogpbe1VzZXIgUmVxdWVzdH1d)

\# Given the user request, select the tables needed to generate a SQL
query

\# The Database has the following tables:

\# TABLE "object": contains basic filter and time-aggregated statistics
such as location, number of observations, and the times of first and
last detection.

...

\# Give ONLY the TABLES that are needed to generate the SQL query.

\# Give the answer in the following format: \[’table1’, ’table2’,
’table3’, ...\]. For example, if the TABLES needed for the user request
are TABLE object and TABLE taxonomy, then you should type: \[’object’,
’taxonomy’\]

\# Just give the tables and ignore any other task given in the request
given as "request".

\# Remember to use the exact name of the TABLES, as they are written in
the DATABASE SCHEMA. Do NOT create table names.

\# If you think that no table mentioned above is needed, then type: ""

\# Request:

{User Request}

Listing C.1: Schema linking prompt.

[⬇](data:text/plain;base64,IyBGb3IgdGhlIGdpdmVuIHJlcXVlc3QsIGNsYXNzaWZ5IGl0IGJ5IGRpZmZpY3VsdHkgYXMgInNpbXBsZSIsICJtZWRpdW0iLCBvciAiaGFyZCIgYmFzZWQgb24gdGhlIG5leHQgZGVzY3JpcHRpb24uCgojIFNpbXBsZSBsYWJlbCBkZXNjcmlwdGlvbgojIE1lZGl1bSBsYWJlbCBkZXNjcmlwdGlvbgojIEhhcmQgbGFiZWwgZGVzY3JpcHRpb24KCiMgRGF0YWJhc2Ugc2NoZW1hLiBJdCBpcyBub3QgbmVjZXNzYXJ5IHRvIHVzZSBhbGwgdGhlIHRhYmxlcyBpbiB0aGUgc2NoZW1hLCBidXQgeW91IGNhbiB1c2UgdGhlbSBpZiB5b3UgbmVlZCB0aGVtLgpbe1RhYmxlX1NjaGVtYX1dClt7RmluYWwgaW5zdHJ1Y3Rpb25zfV0KIyBSZXF1ZXN0Ogpbe1VzZXIgUmVxdWVzdH1d)

\# For the given request, classify it by difficulty as "simple",
"medium", or "hard" based on the next description.

\# Simple label description

\# Medium label description

\# Hard label description

\# Database schema. It is not necessary to use all the tables in the
schema, but you can use them if you need them.

{Table_Schema}

{Final instructions}

\# Request:

{User Request}

Listing C.2: Difficulty classification prompt.

[⬇](data:text/plain;base64,W3tzZWxmX2NvcnJlY3Rpb25fdGFza31dCiMgQ29ycmVjdCBhIFNRTCBxdWVyeSBnaXZlbiB0aGUgbmV4dCB1c2VyIHJlcXVlc3Q6Clt7dXNlcl9yZXF1ZXN0fV0KIyBUaGVzZSBhcmUgdGhlIHRhYmxlIHNjaGVtYXMuIEFzc3VtZSB0aGF0IG9ubHkgdGhlIGZvbGxvd2luZyB0YWJsZXMgYXJlIHJlcXVpcmVkIGZvciB0aGUgcXVlcnk6Clt7dGFibGVfc2NoZW1hc31dCiMgVGhlIGZvbGxvd2luZyBxdWVyeSBpcyBub3Qgd29ya2luZyBkdWUgdG8gYSB0aW1lb3V0IGVycm9yLCBjb3JyZWN0IHRoZSBxdWVyeSB1c2luZyB0aGUgY29ycmVjdCBkYXRhYmFzZSBzY2hlbWEgb3IgbmVzdGVkIHF1ZXJpZXMgdG8gb3B0aW1pemUuCiMgU1FMIFF1ZXJ5Clt7c3FsX3ByZWR9XQojIEVycm9yIHJldHVybmVkIHdoZW4gZXhlY3V0aW5nIHRoZSBxdWVyeSBpbiB0aGUgQUxlUkNFIGRhdGFiYXNlClt7c3FsX2Vycm9yfV0KCiMgRm9sbG93IHRoZXNlIGd1aWRlbGluZXMgdG8gY29ycmVjdCB0aGUgcXVlcnk6CiMgLSBDaGVjayBpZiB0aGUgU1FMIGNvZGUgaW5jbHVkZXMgdGhlIG5lY2Vzc2FyeSBjb25kaXRpb25zIHRvIG9wdGltaXplIHRoZSBxdWVyeSwgYW5kIGlmIHRoZSBxdWVyeSBpcyB1c2luZyB0aGUgY29ycmVjdCBkYXRhYmFzZSBzY2hlbWEgb3IgbmVzdGVkIHF1ZXJpZXMgdG8gb3B0aW1pemUuCiMgICAgIC0gSXQgaXMgcG9zc2libGUgdGhhdCB0aGUgcXVlcnkgaXMgdG9vIGNvbXBsZXggYW5kIHRoYXQgbmVzdGVkIHF1ZXJpZXMgYXJlIG5lY2Vzc2FyeSB0byBvcHRpbWl6ZSBpdC4KIyAgICAgLSBJZiB0aGVyZSBpcyBhIEpPSU4gb3IgYSBzdWItcXVlcnkgYmV0d2VlbiBzb21lIHRhYmxlIGFuZCBwcm9iYWJpbGl0eSwgY2hlY2sgaWYgdGhlIGNvbmRpdGlvbiAncmFua2luZz0xJyBpcyBzZXQgaW4gdGhlIHByb2JhYmlsaXR5IHRhYmxlLCB1bmxlc3MgdGhlIHJlcXVlc3Qgc2FpZCBvdGhlcndpc2UuCiMgLSBJZiB0aGVyZSBhcmUgY29uZGl0aW9ucyBpbnZvbHZpbmcgZGF0ZXMgb3IgdGltZXMsIGNoZWNrIGlmIHRoZSBkYXRlcyBhcmUgbm90IHRvbyBmYXIgYXdheSwgb3IgYXJlIGluIGEgcmVhc29uYWJsZSByYW5nZS4KIyAtIElmIHRoZSBwcm9iYWJpbGl0eSB0YWJsZSBpcyB1c2VkLCBhbHdheXMgdXNlIHRoZSBuZXh0IGNvbmRpdGlvbnMsIHVubGVzcyB0aGUgdXNlciBleHBsaWNpdGx5IHNwZWNpZmllcyBkaWZmZXJlbnQgcHJvYmFiaWxpdHkgY29uZGl0aW9ucy4KIyAtIEVuc3VyZSB0aGVyZSBhcmUgYXQgbGVhc3QgMyBjb25kaXRpb25zIG9uIHRoZSBwcm9iYWJpbGl0eSB0YWJsZSwgYmVjYXVzZSBpZiBub3QsIHRoZSBxdWVyeSBpcyB0b28gZ2VuZXJhbC4gQWRkIG1vcmUgY29uZGl0aW9ucyBpZiBuZWNlc3NhcnkuCgojIENoZWNrIHRoZSBxdWVyeSBhbmQgY29ycmVjdCBpdCBieSBtb2RpZnlpbmcgdGhlIFNRTCBjb2RlIHdoZXJlIHRoZSBlcnJvciBpcyBmb3VuZC4KIyBBZGQgQ09NTUVOVFMgSU4gUG9zdGdyZVNRTCBmb3JtYXQgc28gdGhhdCB0aGUgdXNlciBjYW4gdW5kZXJzdGFuZC4KIyBBbnN3ZXIgT05MWSB3aXRoIHRoZSBTUUwgcXVlcnkKIyBTUUw6)

{self_correction_task}

\# Correct a SQL query given the next user request:

{user_request}

\# These are the table schemas. Assume that only the following tables
are required for the query:

{table_schemas}

\# The following query is not working due to a timeout error, correct
the query using the correct database schema or nested queries to
optimize.

\# SQL Query

{sql_pred}

\# Error returned when executing the query in the ALeRCE database

{sql_error}

\# Follow these guidelines to correct the query:

\# - Check if the SQL code includes the necessary conditions to optimize
the query, and if the query is using the correct database schema or
nested queries to optimize.

\# - It is possible that the query is too complex and that nested
queries are necessary to optimize it.

\# - If there is a JOIN or a sub-query between some table and
probability, check if the condition ’ranking=1’ is set in the
probability table, unless the request said otherwise.

\# - If there are conditions involving dates or times, check if the
dates are not too far away, or are in a reasonable range.

\# - If the probability table is used, always use the next conditions,
unless the user explicitly specifies different probability conditions.

\# - Ensure there are at least 3 conditions on the probability table,
because if not, the query is too general. Add more conditions if
necessary.

\# Check the query and correct it by modifying the SQL code where the
error is found.

\# Add COMMENTS IN PostgreSQL format so that the user can understand.

\# Answer ONLY with the SQL query

\# SQL:

Listing C.3: Example of timeout self-correction prompt.

## Appendix D Perfect match rates by LLM

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | Simple |  | Medium |  | Hard |  |
| Model | w/o Self-Corr | w/ Self-Corr | w/o Self-Corr | w/ Self-Corr | w/o Self-Corr | w/ Self-Corr |
| ID Match (Rows) – Direct |  |  |  |  |  |  |
| Claude 3.7 | 0.844 $`\pm`$ 0.0 | 0.866 $`\pm`$ 0.021 | 0.160 $`\pm`$ 0.084 | 0.160 $`\pm`$ 0.084 | 0.280 $`\pm`$ 0.079 | 0.380 $`\pm`$ 0.079 |
| Claude Opus 4.6 | 0.844 $`\pm`$ 0.0 | 0.938 $`\pm`$ 0.0 | 0.250 $`\pm`$ 0.053 | 0.400 $`\pm`$ 0.047 | 0.400 $`\pm`$ 0.0 | 0.480 $`\pm`$ 0.042 |
| Claude Sonnet 4.5 | 0.844 $`\pm`$ 0.0 | 0.875 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.500 $`\pm`$ 0.0 | 0.200 $`\pm`$ 0.0 | 0.200 $`\pm`$ 0.0 |
| Gemini 2.5 Flash | 0.775 $`\pm`$ 0.020 | 0.819 $`\pm`$ 0.025 | 0.410 $`\pm`$ 0.057 | 0.410 $`\pm`$ 0.057 | 0.240 $`\pm`$ 0.052 | 0.300 $`\pm`$ 5.9e-17 |
| Gemini 2.5 Pro | 0.809 $`\pm`$ 0.023 | 0.856 $`\pm`$ 0.022 | 0.360 $`\pm`$ 0.117 | 0.420 $`\pm`$ 0.079 | 0.160 $`\pm`$ 0.070 | 0.360 $`\pm`$ 0.052 |
| Gemini 3 Flash | 0.866 $`\pm`$ 0.015 | 0.928 $`\pm`$ 0.015 | 0.300 $`\pm`$ 5.9e-17 | 0.400 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.500 $`\pm`$ 0.0 |
| Gemini 3.1 Flash | 0.844 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.200 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 |
| GPT-4.1 | 0.753 $`\pm`$ 0.023 | 0.806 $`\pm`$ 0.029 | 0.210 $`\pm`$ 0.088 | 0.260 $`\pm`$ 0.070 | 0.270 $`\pm`$ 0.048 | 0.340 $`\pm`$ 0.052 |
| GPT-4o | 0.856 $`\pm`$ 0.016 | 0.891 $`\pm`$ 0.022 | 0.130 $`\pm`$ 0.048 | 0.260 $`\pm`$ 0.084 | 0.0 $`\pm`$ 0.0 | 0.110 $`\pm`$ 0.032 |
| GPT-5 | 0.787 $`\pm`$ 0.032 | 0.822 $`\pm`$ 0.033 | 0.350 $`\pm`$ 0.085 | 0.380 $`\pm`$ 0.063 | 0.230 $`\pm`$ 0.048 | 0.290 $`\pm`$ 0.032 |
| GPT-5.2 | 0.784 $`\pm`$ 0.023 | 0.800 $`\pm`$ 0.026 | 0.280 $`\pm`$ 0.063 | 0.300 $`\pm`$ 0.067 | 0.370 $`\pm`$ 0.116 | 0.380 $`\pm`$ 0.132 |
| GPT-5.2-Codex | 0.731 $`\pm`$ 0.030 | 0.816 $`\pm`$ 0.023 | 0.320 $`\pm`$ 0.092 | 0.340 $`\pm`$ 0.084 | 0.280 $`\pm`$ 0.063 | 0.310 $`\pm`$ 0.032 |
| GPT-5.3-Codex | 0.753 $`\pm`$ 0.018 | 0.775 $`\pm`$ 0.025 | 0.200 $`\pm`$ 0.067 | 0.210 $`\pm`$ 0.074 | 0.190 $`\pm`$ 0.032 | 0.330 $`\pm`$ 0.048 |
| ID Match (Rows) – Step-by-Step |  |  |  |  |  |  |
| Claude 3.7 | 0.794 $`\pm`$ 0.016 | 0.828 $`\pm`$ 0.030 | 0.230 $`\pm`$ 0.177 | 0.250 $`\pm`$ 0.165 | 0.150 $`\pm`$ 0.085 | 0.250 $`\pm`$ 0.085 |
| Claude Opus 4.6 | 0.938 $`\pm`$ 0.0 | 0.969 $`\pm`$ 0.0 | 0.380 $`\pm`$ 0.042 | 0.440 $`\pm`$ 0.070 | 0.530 $`\pm`$ 0.048 | 0.590 $`\pm`$ 0.032 |
| Claude Sonnet 4.5 | 0.906 $`\pm`$ 0.0 | 0.938 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.200 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 |
| Gemini 2.5 Flash | 0.853 $`\pm`$ 0.015 | 0.853 $`\pm`$ 0.015 | 0.200 $`\pm`$ 0.0 | 0.250 $`\pm`$ 0.053 | 0.190 $`\pm`$ 0.074 | 0.230 $`\pm`$ 0.048 |
| Gemini 2.5 Pro | 0.847 $`\pm`$ 0.023 | 0.875 $`\pm`$ 0.015 | 0.490 $`\pm`$ 0.032 | 0.500 $`\pm`$ 0.0 | 0.250 $`\pm`$ 0.053 | 0.370 $`\pm`$ 0.048 |
| Gemini 3 Flash | 0.938 $`\pm`$ 0.0 | 0.938 $`\pm`$ 0.0 | 0.390 $`\pm`$ 0.032 | 0.400 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 | 0.400 $`\pm`$ 0.0 |
| Gemini 3.1 Flash | 0.812 $`\pm`$ 0.0 | 0.844 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 | 0.300 $`\pm`$ 5.9e-17 | 0.100 $`\pm`$ 0.0 | 0.200 $`\pm`$ 0.0 |
| GPT-4.1 | 0.797 $`\pm`$ 0.022 | 0.809 $`\pm`$ 0.034 | 0.270 $`\pm`$ 0.082 | 0.300 $`\pm`$ 0.115 | 0.210 $`\pm`$ 0.074 | 0.280 $`\pm`$ 0.079 |
| GPT-4o | 0.875 $`\pm`$ 0.033 | 0.900 $`\pm`$ 0.029 | 0.180 $`\pm`$ 0.042 | 0.270 $`\pm`$ 0.048 | 0.130 $`\pm`$ 0.067 | 0.220 $`\pm`$ 0.063 |
| GPT-5 | 0.856 $`\pm`$ 0.030 | 0.884 $`\pm`$ 0.039 | 0.370 $`\pm`$ 0.048 | 0.390 $`\pm`$ 0.032 | 0.190 $`\pm`$ 0.032 | 0.260 $`\pm`$ 0.070 |
| GPT-5.2 | 0.806 $`\pm`$ 0.029 | 0.809 $`\pm`$ 0.031 | 0.280 $`\pm`$ 0.042 | 0.290 $`\pm`$ 0.032 | 0.290 $`\pm`$ 0.074 | 0.290 $`\pm`$ 0.074 |
| GPT-5.2-Codex | 0.847 $`\pm`$ 0.045 | 0.869 $`\pm`$ 0.041 | 0.420 $`\pm`$ 0.092 | 0.440 $`\pm`$ 0.107 | 0.250 $`\pm`$ 0.053 | 0.280 $`\pm`$ 0.079 |
| GPT-5.3-Codex | 0.838 $`\pm`$ 0.013 | 0.875 $`\pm`$ 0.021 | 0.310 $`\pm`$ 0.074 | 0.320 $`\pm`$ 0.063 | 0.350 $`\pm`$ 0.053 | 0.360 $`\pm`$ 0.070 |
| Column Match – Direct |  |  |  |  |  |  |
| Claude 3.7 | 0.738 $`\pm`$ 0.030 | 0.762 $`\pm`$ 0.034 | 0.680 $`\pm`$ 0.042 | 0.790 $`\pm`$ 0.057 | 0.310 $`\pm`$ 0.088 | 0.280 $`\pm`$ 0.092 |
| Claude Opus 4.6 | 0.797 $`\pm`$ 0.016 | 0.812 $`\pm`$ 0.0 | 0.640 $`\pm`$ 0.052 | 0.790 $`\pm`$ 0.032 | 0.220 $`\pm`$ 0.042 | 0.220 $`\pm`$ 0.042 |
| Claude Sonnet 4.5 | 0.781 $`\pm`$ 0.0 | 0.781 $`\pm`$ 0.0 | 0.500 $`\pm`$ 0.0 | 0.700 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 | 0.300 $`\pm`$ 5.9e-17 |
| Gemini 2.5 Flash | 0.775 $`\pm`$ 0.025 | 0.800 $`\pm`$ 0.022 | 0.760 $`\pm`$ 0.052 | 0.820 $`\pm`$ 0.079 | 0.240 $`\pm`$ 0.052 | 0.350 $`\pm`$ 0.071 |
| Gemini 2.5 Pro | 0.828 $`\pm`$ 0.016 | 0.828 $`\pm`$ 0.016 | 0.760 $`\pm`$ 0.097 | 0.820 $`\pm`$ 0.063 | 0.200 $`\pm`$ 0.067 | 0.280 $`\pm`$ 0.042 |
| Gemini 3 Flash | 0.803 $`\pm`$ 0.015 | 0.812 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.400 $`\pm`$ 0.0 | 0.500 $`\pm`$ 0.0 |
| Gemini 3.1 Flash | 0.812 $`\pm`$ 0.0 | 0.812 $`\pm`$ 0.0 | 0.900 $`\pm`$ 0.0 | 0.900 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 | 0.300 $`\pm`$ 5.9e-17 |
| GPT-4.1 | 0.709 $`\pm`$ 0.015 | 0.741 $`\pm`$ 0.015 | 0.600 $`\pm`$ 0.047 | 0.650 $`\pm`$ 0.053 | 0.230 $`\pm`$ 0.048 | 0.230 $`\pm`$ 0.048 |
| GPT-4o | 0.728 $`\pm`$ 0.026 | 0.738 $`\pm`$ 0.016 | 0.270 $`\pm`$ 0.082 | 0.450 $`\pm`$ 0.085 | 0.100 $`\pm`$ 0.0 | 0.120 $`\pm`$ 0.042 |
| GPT-5 | 0.750 $`\pm`$ 0.026 | 0.775 $`\pm`$ 0.025 | 0.690 $`\pm`$ 0.110 | 0.760 $`\pm`$ 0.107 | 0.250 $`\pm`$ 0.071 | 0.280 $`\pm`$ 0.063 |
| GPT-5.2 | 0.762 $`\pm`$ 0.034 | 0.778 $`\pm`$ 0.031 | 0.750 $`\pm`$ 0.071 | 0.770 $`\pm`$ 0.048 | 0.270 $`\pm`$ 0.067 | 0.300 $`\pm`$ 0.067 |
| GPT-5.2-Codex | 0.728 $`\pm`$ 0.033 | 0.809 $`\pm`$ 0.031 | 0.760 $`\pm`$ 0.084 | 0.790 $`\pm`$ 0.088 | 0.280 $`\pm`$ 0.042 | 0.320 $`\pm`$ 0.042 |
| GPT-5.3-Codex | 0.759 $`\pm`$ 0.030 | 0.797 $`\pm`$ 0.022 | 0.560 $`\pm`$ 0.052 | 0.560 $`\pm`$ 0.052 | 0.190 $`\pm`$ 0.057 | 0.330 $`\pm`$ 0.048 |
| Column Match – Step-by-Step |  |  |  |  |  |  |
| Claude 3.7 | 0.734 $`\pm`$ 0.034 | 0.778 $`\pm`$ 0.027 | 0.530 $`\pm`$ 0.149 | 0.660 $`\pm`$ 0.143 | 0.280 $`\pm`$ 0.103 | 0.260 $`\pm`$ 0.052 |
| Claude Opus 4.6 | 0.938 $`\pm`$ 0.0 | 0.938 $`\pm`$ 0.0 | 0.650 $`\pm`$ 0.053 | 0.720 $`\pm`$ 0.042 | 0.300 $`\pm`$ 5.9e-17 | 0.310 $`\pm`$ 0.032 |
| Claude Sonnet 4.5 | 0.875 $`\pm`$ 0.0 | 0.875 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.900 $`\pm`$ 0.0 | 0.250 $`\pm`$ 0.053 | 0.300 $`\pm`$ 5.9e-17 |
| Gemini 2.5 Flash | 0.906 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.850 $`\pm`$ 0.053 | 0.880 $`\pm`$ 0.042 | 0.200 $`\pm`$ 0.0 | 0.240 $`\pm`$ 0.052 |
| Gemini 2.5 Pro | 0.906 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.790 $`\pm`$ 0.032 | 0.800 $`\pm`$ 0.047 | 0.240 $`\pm`$ 0.052 | 0.310 $`\pm`$ 0.057 |
| Gemini 3 Flash | 0.906 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.810 $`\pm`$ 0.032 | 0.300 $`\pm`$ 5.9e-17 | 0.300 $`\pm`$ 5.9e-17 |
| Gemini 3.1 Flash | 0.875 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.800 $`\pm`$ 0.0 | 0.300 $`\pm`$ 5.9e-17 | 0.300 $`\pm`$ 5.9e-17 |
| GPT-4.1 | 0.903 $`\pm`$ 0.010 | 0.903 $`\pm`$ 0.010 | 0.680 $`\pm`$ 0.063 | 0.750 $`\pm`$ 0.053 | 0.250 $`\pm`$ 0.053 | 0.290 $`\pm`$ 0.032 |
| GPT-4o | 0.822 $`\pm`$ 0.021 | 0.822 $`\pm`$ 0.021 | 0.540 $`\pm`$ 0.097 | 0.700 $`\pm`$ 0.094 | 0.360 $`\pm`$ 0.070 | 0.390 $`\pm`$ 0.074 |
| GPT-5 | 0.856 $`\pm`$ 0.037 | 0.872 $`\pm`$ 0.031 | 0.750 $`\pm`$ 0.097 | 0.830 $`\pm`$ 0.067 | 0.050 $`\pm`$ 0.053 | 0.180 $`\pm`$ 0.092 |
| GPT-5.2 | 0.863 $`\pm`$ 0.016 | 0.863 $`\pm`$ 0.016 | 0.760 $`\pm`$ 0.052 | 0.770 $`\pm`$ 0.067 | 0.390 $`\pm`$ 0.074 | 0.390 $`\pm`$ 0.074 |
| GPT-5.2-Codex | 0.887 $`\pm`$ 0.037 | 0.900 $`\pm`$ 0.029 | 0.810 $`\pm`$ 0.057 | 0.830 $`\pm`$ 0.048 | 0.330 $`\pm`$ 0.067 | 0.380 $`\pm`$ 0.042 |
| GPT-5.3-Codex | 0.875 $`\pm`$ 0.0 | 0.906 $`\pm`$ 0.0 | 0.690 $`\pm`$ 0.057 | 0.730 $`\pm`$ 0.067 | 0.370 $`\pm`$ 0.067 | 0.380 $`\pm`$ 0.079 |

Table 9: Perfect match rates by LLM, query generation method, query
difficulty, and w/ or w/o self-correction.²¹²¹ 21 Notes. In each of the
four sections of the table, the best mean per column is shown in
boldface type, together with every LLM model whose run-level scores are
not significantly worse than the best under a two-sided unpaired
permutation test (Holm-corrected, $`\alpha=0.05`$).

## Appendix E Additional results

| Model | Direct |  |  |  |  |  | Step-by-Step |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | Total | Schema | Difficulty | Plan | SQL | Self-Correction | Total | Schema | Difficulty | Plan | SQL | Self-Correction |
| GPT-5.2 | 42 671 | 802 | 0 | 0 | 15 363 | 28 877 | 19 357 | 802 | 4 328 | 8 554 | 7 991 | 5 303 |
| GPT-5.3-Codex | 39 907 | 801 | 0 | 0 | 12 261 | 27 044 | 22 978 | 801 | 4 263 | 7 039 | 6 574 | 6 712 |
| GPT-5.2-Codex | 22 106 | 1 539 | 0 | 0 | 13 740 | 9 029 | 35 236 | 1 539 | 5 085 | 12 228 | 10 697 | 9 252 |
| GPT-5 | 26 655 | 3 919 | 0 | 0 | 13 312 | 11 232 | 58 071 | 3 919 | 6 609 | 20 935 | 16 627 | 17 157 |
| Claude Opus 4.6 | 8 193 | 1 122 | 0 | 0 | 4 716 | 4 135 | 13 439 | 1 122 | 3 648 | 5 524 | 4 415 | 8 098 |
| GPT-4.1 | 20 738 | 795 | 0 | 0 | 8 659 | 11 351 | 20 279 | 951 | 4 201 | 7 557 | 6 878 | 5 160 |
| Claude Sonnet 3.7 | 11 303 | 2 091 | 0 | 0 | 1 211 | 8 138 | 8 243 | 2 091 | 1 190 | 1 162 | 1 153 | 4 543 |
| Gemini 2.5 Pro | 20 741 | 7 936 | 0 | 0 | 6 826 | 6 579 | 34 344 | 7 936 | 4 606 | 8 128 | 7 499 | 6 795 |
| GPT-4o | 9 509 | 795 | 0 | 0 | 6 632 | 4 127 | 18 677 | 795 | 4 809 | 7 622 | 4 647 | 5 010 |
| Claude Sonnet 4.5 | 5 737 | 1 729 | 0 | 0 | 947 | 3 692 | 12 701 | 1 729 | 1 194 | 2 088 | 1 101 | 7 872 |
| Gemini 3 Flash | 8 557 | 1 924 | 0 | 0 | 6 084 | 3 437 | 18 789 | 1 924 | 4 486 | 7 458 | 6 696 | 6 595 |
| Gemini 3.1 Flash | 8 206 | 913 | 0 | 0 | 7 067 | 3 397 | 27 134 | 913 | 5 144 | 7 835 | 7 982 | 5 349 |

Table 10: Total and stagewise maximum per-query token counts by LLM
model and query generation method.²²²² 22 Notes. Values represent the
highest token count observed across all evaluated queries for each
stage. The largest value per column is bolded to indicate the upper
bound.

![Refer to
caption](2606.18108v1/results_llm_claude_comparison_v6_direct.png)

(a) Direct generation method

![Refer to
caption](2606.18108v1/results_llm_claude_comparison_v6_sbs.png)

(b) Step-by-step generation method

Figure 12: Performance comparison of Claude models, as a function of the
level of query difficulty.

![Refer to
caption](2606.18108v1/results_llm_gemini_comparison_v6_direct.png)

(a) Direct generation method

![Refer to
caption](2606.18108v1/results_llm_gemini_comparison_v6_sbs.png)

(b) Step-by-step generation method

Figure 13: Performance comparison of Gemini models, as a function of the
level of query difficulty.

![Refer to
caption](2606.18108v1/results_llm_gpt_comparison_v6_direct.png)

(a) Direct generation method

![Refer to caption](2606.18108v1/results_llm_gpt_comparison_v6_sbs.png)

(b) Step-by-step generation method

Figure 14: Performance comparison of GPT models, as a function of the
level of query difficulty.

![Refer to caption](2606.18108v1/results_nosc_vs_sc_v2_gpt52codex.png)

Figure 15: GPT-5.2-Codex performance without vs with self correction for
direct and step-by-step generation methods.

![Refer to
caption](2606.18108v1/percentage_error_difficulty_gpt52codex.png)

Figure 16: Percentage of perfect queries, semantic errors, and execution
errors for the direct and step-by-step query generation methods using
GPT-5.2-Codex.

![Refer to
caption](2606.18108v1/percentage_error_difficulty_gemini3flash.png)

Figure 17: Number of perfect queries, semantic errors, and execution
errors for the direct and step-by-step query generation methods using
Gemini 3 Flash.

## Appendix F Few-shot prompting

|  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Few-shot | Selection | Gemini 2.5 Pro |  |  |  |  |  |
|  |  | Simple |  | Medium |  | Hard |  |
|  |  | Rows | Columns | Rows | Columns | Rows | Columns |
| 0-shot | - | 0.88 $`\pm`$ 0.015 | 0.88 $`\pm`$ 0.015 | 0.50 $`\pm`$ 0.000 | 0.76 $`\pm`$ 0.084 | 0.37 $`\pm`$ 0.048 | 0.36 $`\pm`$ 0.084 |
| 1-shot | Random | 0.97 $`\pm`$ 0.000 $`\uparrow`$ | 0.92 $`\pm`$ 0.017 $`\sim`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ | 0.82 $`\pm`$ 0.084 $`\sim`$ | 0.32 $`\pm`$ 0.045 $`\downarrow`$ | 0.40 $`\pm`$ 0.122 $`\sim`$ |
|  | QTS | 0.96 $`\pm`$ 0.017 $`\uparrow`$ | 0.94 $`\pm`$ 0.014 $`\uparrow`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ | 0.74 $`\pm`$ 0.055 $`\sim`$ | 0.36 $`\pm`$ 0.114 $`\sim`$ | 0.38 $`\pm`$ 0.084 $`\sim`$ |
|  | MQS | 0.95 $`\pm`$ 0.017 $`\uparrow`$ | 0.93 $`\pm`$ 0.017 $`\uparrow`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ | 0.82 $`\pm`$ 0.045 $`\sim`$ | 0.34 $`\pm`$ 0.055 $`\sim`$ | 0.46 $`\pm`$ 0.055 $`\sim`$ |
| 3-shot | Random | 0.95 $`\pm`$ 0.028 $`\uparrow`$ | 0.94 $`\pm`$ 0.000 $`\uparrow`$ | 0.60 $`\pm`$ 0.000 $`\uparrow`$ | 0.82 $`\pm`$ 0.045 $`\sim`$ | 0.40 $`\pm`$ 0.122 $`\sim`$ | 0.42 $`\pm`$ 0.045 $`\sim`$ |
|  | QTS | 0.95 $`\pm`$ 0.017 $`\uparrow`$ | 0.95 $`\pm`$ 0.017 $`\uparrow`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ | 0.84 $`\pm`$ 0.055 $`\sim`$ | 0.48 $`\pm`$ 0.045 $`\uparrow`$ | 0.46 $`\pm`$ 0.055 $`\uparrow`$ |
|  | MQS | 0.94 $`\pm`$ 0.014 $`\uparrow`$ | 0.94 $`\pm`$ 0.000 $`\uparrow`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ | 0.82 $`\pm`$ 0.045 $`\sim`$ | 0.32 $`\pm`$ 0.045 $`\downarrow`$ | 0.36 $`\pm`$ 0.055 $`\sim`$ |

Table 11: Few-shot PM results by query difficulty when using Gemini 2.5
Pro (step-by-step).²³²³ 23 Notes. The best value per table column is
bolded. Arrows compare the few-shot setting against the 0-shot baseline
for each query difficulty using two-sided permutation tests on
perfect-match scores grouped by run; $`\uparrow`$/$`\downarrow`$
indicate significantly higher/lower results at $`\alpha=0.05`$ with no
multiple-testing correction, and $`\sim`$ indicates not significant.

|  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| Few-shot | Selection | Claude Opus 4.6 |  |  |  |  |  |
|  |  | Simple |  | Medium |  | Hard |  |
|  |  | Rows | Columns | Rows | Columns | Rows | Columns |
| 0-shot | - | 0.97 $`\pm`$ 0.000 | 0.94 $`\pm`$ 0.000 | 0.44 $`\pm`$ 0.070 | 0.72 $`\pm`$ 0.042 | 0.59 $`\pm`$ 0.032 | 0.49 $`\pm`$ 0.057 |
| 1-shot | Random | 0.97 $`\pm`$ 0.000 $`\sim`$ | 0.93 $`\pm`$ 0.017 $`\sim`$ | 0.32 $`\pm`$ 0.045 $`\downarrow`$ | 0.72 $`\pm`$ 0.045 $`\sim`$ | 0.42 $`\pm`$ 0.045 $`\downarrow`$ | 0.46 $`\pm`$ 0.055 $`\sim`$ |
|  | QTS | 0.97 $`\pm`$ 0.000 $`\sim`$ | 0.94 $`\pm`$ 0.000 $`\sim`$ | 0.30 $`\pm`$ 0.000 $`\downarrow`$ | 0.70 $`\pm`$ 0.000 $`\sim`$ | 0.48 $`\pm`$ 0.045 $`\downarrow`$ | 0.50 $`\pm`$ 0.000 $`\sim`$ |
|  | MQS | 0.97 $`\pm`$ 0.000 $`\sim`$ | 0.96 $`\pm`$ 0.017 $`\sim`$ | 0.42 $`\pm`$ 0.045 $`\sim`$ | 0.82 $`\pm`$ 0.045 $`\uparrow`$ | 0.52 $`\pm`$ 0.045 $`\downarrow`$ | 0.48 $`\pm`$ 0.045 $`\sim`$ |
| 3-shot | Random | 0.90 $`\pm`$ 0.014 $`\downarrow`$ | 0.88 $`\pm`$ 0.014 $`\downarrow`$ | 0.38 $`\pm`$ 0.084 $`\sim`$ | 0.82 $`\pm`$ 0.045 $`\uparrow`$ | 0.38 $`\pm`$ 0.045 $`\downarrow`$ | 0.42 $`\pm`$ 0.084 $`\sim`$ |
|  | QTS | 0.97 $`\pm`$ 0.014 $`\sim`$ | 0.97 $`\pm`$ 0.000 $`\uparrow`$ | 0.40 $`\pm`$ 0.000 $`\sim`$ | 0.70 $`\pm`$ 0.000 $`\sim`$ | 0.48 $`\pm`$ 0.045 $`\downarrow`$ | 0.48 $`\pm`$ 0.045 $`\sim`$ |
|  | MQS | 0.94 $`\pm`$ 0.000 $`\downarrow`$ | 0.94 $`\pm`$ 0.000 $`\sim`$ | 0.30 $`\pm`$ 0.000 $`\downarrow`$ | 0.70 $`\pm`$ 0.000 $`\sim`$ | 0.58 $`\pm`$ 0.045 $`\sim`$ | 0.46 $`\pm`$ 0.055 $`\sim`$ |

Table 12: Few-shot PM results by query difficulty when using Claude Opus
4.6 (step-by-step).²⁴²⁴ 24 Notes. The best value per table column is
bolded. Arrows compare the few-shot setting against the 0-shot baseline
for each query difficulty using two-sided permutation tests on
perfect-match scores grouped by run; $`\uparrow`$/$`\downarrow`$
indicate significantly higher/lower results at $`\alpha=0.05`$ with no
multiple-testing correction, and $`\sim`$ indicates not significant.

[⬇](data:text/plain;base64,IyMgRXhhbXBsZSAxICh0YWJsZXM6IGRhdGFxdWFsaXR5LCAuLi4pCiMgSW1wb3J0YW50IEluZm9ybWF0aW9uIGZvciB0aGUgcXVlcnkKRXh0ZXJuYWwgS25vd2xlZGdlOgojIFVzZXIgUmVxdWVzdDogLi4uCmBgYHNxbAouLi4KYGBgCiMjIEVuZCBFeGFtcGxlCgojIEltcG9ydGFudCBJbmZvcm1hdGlvbiBmb3IgdGhlIHF1ZXJ5CkV4dGVybmFsIEtub3dsZWRnZToKIyBVc2VyIFJlcXVlc3Q6)

\## Example 1 (tables: dataquality, ...)

\# Important Information for the query

External Knowledge:

\# User Request: ...

‘‘‘sql

...

‘‘‘

\## End Example

\# Important Information for the query

External Knowledge:

\# User Request:

Listing F.1: Example few-shot prompt

## Appendix G Function calling and structured outputs

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | Simple |  | Medium |  | Hard |  |
| Model | w/o Func. call | w/ Func. call | w/o Func. call | w/ Func. call | w/o Func. call | w/ Func. call |
| ID Match (Rows) – Direct |  |  |  |  |  |  |
| Claude Opus 4.6 | 0.844 $`\pm`$ 0.000 | 0.850 $`\pm`$ 0.014 | 0.250 $`\pm`$ 0.053 | 0.280 $`\pm`$ 0.045 | 0.400 $`\pm`$ 0.000 | 0.460 $`\pm`$ 0.055 |
| Gemini 2.5 Pro | 0.809 $`\pm`$ 0.023 | 0.787 $`\pm`$ 0.014 | 0.360 $`\pm`$ 0.117 | 0.320 $`\pm`$ 0.045 | 0.160 $`\pm`$ 0.070 | 0.200 $`\pm`$ 0.071 |
| Gemini 3 Flash | 0.866 $`\pm`$ 0.015 | 0.769 $`\pm`$ 0.034 | 0.300 $`\pm`$ 0.000 | 0.320 $`\pm`$ 0.042 | 0.300 $`\pm`$ 0.000 | 0.360 $`\pm`$ 0.052 |
| ID Match (Rows) – Step-by-Step |  |  |  |  |  |  |
| Claude Opus 4.6 | 0.938 $`\pm`$ 0.000 | 0.938 $`\pm`$ 0.000 | 0.380 $`\pm`$ 0.042 | 0.390 $`\pm`$ 0.088 | 0.530 $`\pm`$ 0.048 | 0.460 $`\pm`$ 0.084 |
| Gemini 2.5 Pro | 0.847 $`\pm`$ 0.023 | 0.863 $`\pm`$ 0.017 | 0.490 $`\pm`$ 0.032 | 0.500 $`\pm`$ 0.000 | 0.250 $`\pm`$ 0.053 | 0.240 $`\pm`$ 0.055 |
| Gemini 3 Flash | 0.969 $`\pm`$ 0.000 | 0.944 $`\pm`$ 0.025 | 0.290 $`\pm`$ 0.032 | 0.310 $`\pm`$ 0.088 | 0.300 $`\pm`$ 0.000 | 0.240 $`\pm`$ 0.052 |
| Column Match – Direct |  |  |  |  |  |  |
| Claude Opus 4.6 | 0.797 $`\pm`$ 0.016 | 0.787 $`\pm`$ 0.014 | 0.640 $`\pm`$ 0.052 | 0.600 $`\pm`$ 0.071 | 0.320 $`\pm`$ 0.042 | 0.300 $`\pm`$ 0.000 |
| Gemini 2.5 Pro | 0.797 $`\pm`$ 0.016 | 0.831 $`\pm`$ 0.017 | 0.660 $`\pm`$ 0.084 | 0.560 $`\pm`$ 0.089 | 0.180 $`\pm`$ 0.079 | 0.220 $`\pm`$ 0.045 |
| Gemini 3 Flash | 0.803 $`\pm`$ 0.015 | 0.806 $`\pm`$ 0.020 | 0.800 $`\pm`$ 0.000 | 0.710 $`\pm`$ 0.057 | 0.400 $`\pm`$ 0.000 | 0.420 $`\pm`$ 0.042 |
| Column Match – Step-by-Step |  |  |  |  |  |  |
| Claude Opus 4.6 | 0.938 $`\pm`$ 0.000 | 0.938 $`\pm`$ 0.000 | 0.650 $`\pm`$ 0.053 | 0.660 $`\pm`$ 0.126 | 0.480 $`\pm`$ 0.042 | 0.460 $`\pm`$ 0.052 |
| Gemini 2.5 Pro | 0.884 $`\pm`$ 0.015 | 0.869 $`\pm`$ 0.026 | 0.750 $`\pm`$ 0.071 | 0.780 $`\pm`$ 0.084 | 0.290 $`\pm`$ 0.057 | 0.260 $`\pm`$ 0.055 |
| Gemini 3 Flash | 0.938 $`\pm`$ 0.000 | 0.938 $`\pm`$ 0.000 | 0.800 $`\pm`$ 0.000 | 0.830 $`\pm`$ 0.095 | 0.400 $`\pm`$ 0.000 | 0.420 $`\pm`$ 0.042 |

Table 13: Perfect match rates by LLM, query generation method, query
difficulty, and w/ or w/o function calling (w/o self-correction).²⁵²⁵ 25
Notes. Best results per column in each of the four sections of the table
are bolded.

|  |  |  |  |  |
|----|----|----|----|----|
|  | Medium |  | Hard |  |
| Model | Free-form | Structured outputs | Free-form | Structured outputs |
| Rows |  |  |  |  |
| Gemini 3 Flash | 0.40 $`\pm`$ 0.000 | 0.30 $`\pm`$ 0.100 $`\downarrow`$ | 0.40 $`\pm`$ 0.000 | 0.38 $`\pm`$ 0.045 $`\sim`$ |
| Gemini 2.5 Pro | 0.50 $`\pm`$ 0.000 | 0.50 $`\pm`$ 0.000 | 0.37 $`\pm`$ 0.048 | 0.34 $`\pm`$ 0.055 $`\sim`$ |
| Claude Opus 4.6 | 0.44 $`\pm`$ 0.070 | 0.30 $`\pm`$ 0.000 $`\downarrow`$ | 0.59 $`\pm`$ 0.032 | 0.58 $`\pm`$ 0.130 $`\sim`$ |
| Columns |  |  |  |  |
| Gemini 3 Flash | 0.81 $`\pm`$ 0.032 | 0.88 $`\pm`$ 0.045 $`\uparrow`$ | 0.30 $`\pm`$ 0.000 | 0.30 $`\pm`$ 0.000 |
| Gemini 2.5 Pro | 0.80 $`\pm`$ 0.047 | 0.84 $`\pm`$ 0.055 $`\sim`$ | 0.31 $`\pm`$ 0.057 | 0.28 $`\pm`$ 0.045 $`\sim`$ |
| Claude Opus 4.6 | 0.72 $`\pm`$ 0.042 | 0.70 $`\pm`$ 0.000 $`\sim`$ | 0.31 $`\pm`$ 0.032 | 0.28 $`\pm`$ 0.045 $`\sim`$ |

Table 14: Structured vs. base step-by-step plan comparison, by
difficulty (with self-correction).²⁶²⁶ 26 Notes. Each pair shows rows
and columns as mean $`\pm`$ std over run-level means; best per column is
bolded. Markers on structured cells reflect a unpaired permutation test,
$`\uparrow`$/$`\downarrow`$ when the adjusted $`p`$-value rejects
equality, $`\sim`$ otherwise.

[⬇](data:text/plain;base64,ZGF0ZV90b19tamRfdG9vbCA9IHR5cGVzLkZ1bmN0aW9uRGVjbGFyYXRpb24oCiAgICAgICAgICAgIG5hbWU9ImRhdGVfdG9fbWpkIiwKICAgICAgICAgICAgZGVzY3JpcHRpb249KAogICAgICAgICAgICAgICAgIkNvbnZlcnQgT05FIGRhdGUgZXhwcmVzc2lvbiB0byBhIHNpbmdsZSBNSkQgdmFsdWUuIEFjY2VwdHMgSVNPIDg2MDEgKGUuZy4gJzIwMjItMTItMDEnKSwgb3IgcmVsYXRpdmUgZXhwcmVzc2lvbnMgYW5jaG9yZWQgdG8gdGhlIGN1cnJlbnQgdGltZSAoZS5nLiAnMzAwIGRheXMgYWdvJywgJzcgZGF5cyBhZ28nKS4gIgogICAgICAgICAgICAgICAgIkZvciBkYXRlIHJhbmdlcywgY2FsbCB0aGlzIHRvb2wgb25jZSBwZXIgYm91bmRhcnkuICIKICAgICAgICAgICAgICAgICJBTGVSQ0Ugc3RvcmVzIGFsbCBkYXRlcyBhcyBNSkQuIgogICAgICAgICAgICApLAogICAgICAgICAgICBwYXJhbWV0ZXJzPXR5cGVzLlNjaGVtYSgKICAgICAgICAgICAgICAgIHR5cGU9dHlwZXMuVHlwZS5PQkpFQ1QsCiAgICAgICAgICAgICAgICBwcm9wZXJ0aWVzPXsKICAgICAgICAgICAgICAgICAgICAiZGF0ZV9leHByZXNzaW9uIjogdHlwZXMuU2NoZW1hKAogICAgICAgICAgICAgICAgICAgICAgICB0eXBlPXR5cGVzLlR5cGUuU1RSSU5HLAogICAgICAgICAgICAgICAgICAgICAgICBkZXNjcmlwdGlvbj0iQSBzaW5nbGUgZGF0ZTogSVNPIDg2MDEgKCcyMDIzLTA5LTAxJykgb3IgcmVsYXRpdmUgKCczMDAgZGF5cyBhZ28nLCAnNyBkYXlzIGFnbycpLiIKICAgICAgICAgICAgICAgICAgICApLAogICAgICAgICAgICAgICAgICAgICJsYWJlbCI6IHR5cGVzLlNjaGVtYSgKICAgICAgICAgICAgICAgICAgICAgICAgdHlwZT10eXBlcy5UeXBlLlNUUklORywKICAgICAgICAgICAgICAgICAgICAgICAgZGVzY3JpcHRpb249IkEgc2hvcnQgc25ha2VfY2FzZSBrZXkgc3VtbWFyaXppbmcgdGhlIGRhdGUgYXMgbWpkX1ttb250aF1fW2RheV0gb3IgbWpkX1tyZWxhdGl2ZV1fW2NvdW50XSwgZS5nLiAnbWpkX2F1Z3VzdF8yMycgb3IgJ21qZF9ob3Vyc18xJy4iCiAgICAgICAgICAgICAgICAgICAgKSwKICAgICAgICAgICAgICAgIH0sCiAgICAgICAgICAgICAgICByZXF1aXJlZD1bImRhdGVfZXhwcmVzc2lvbiJdLAogICAgICAgICAgICApLAogICAgICAgICk=)

date_to_mjd_tool = types.FunctionDeclaration(

name="date_to_mjd",

description=(

"Convert ONE date expression to a single MJD value. Accepts ISO 8601
(e.g. ’2022-12-01’), or relative expressions anchored to the current
time (e.g. ’300 days ago’, ’7 days ago’). "

"For date ranges, call this tool once per boundary. "

"ALeRCE stores all dates as MJD."

),

parameters=types.Schema(

type=types.Type.OBJECT,

properties={

"date_expression": types.Schema(

type=types.Type.STRING,

description="A single date: ISO 8601 (’2023-09-01’) or relative (’300
days ago’, ’7 days ago’)."

),

"label": types.Schema(

type=types.Type.STRING,

description="A short snake_case key summarizing the date as
mjd\_\[month\]\_\[day\] or mjd\_\[relative\]\_\[count\], e.g.
’mjd_august_23’ or ’mjd_hours_1’."

),

},

required="date_expression",

),

)

Listing G.1: Specialized tool defined in Python to transform date
formats to MJD.

[⬇](data:text/plain;base64,CmNsYXNzIFBsYW5TdGVwKEJhc2VNb2RlbCk6CiAgICBzdGVwOiBzdHIgPSBGaWVsZCgKICAgICAgICBkZXNjcmlwdGlvbj0iQSBzaW5nbGUgZGVjb21wb3NpdGlvbiBzdGVwIGRlc2NyaWJpbmcgcGFydCBvZiB0aGUgcGxhbiB0byBnZW5lcmF0ZSB0aGUgU1FMIHF1ZXJ5LiIpCmNsYXNzIERlY29tcG9zaXRpb25QbGFuT3V0cHV0KEJhc2VNb2RlbCk6CiAgICBzdGVwczogbGlzdFtQbGFuU3RlcF0gPSBGaWVsZCgKICAgICAgICBkZXNjcmlwdGlvbj0iT3JkZXJlZCBkZWNvbXBvc2l0aW9uIHN0ZXBzIGZvciB0aGUgU1FMIGdlbmVyYXRpb24gcGxhbi4iLAogICAgICAgIG1pbl9sZW5ndGg9MSwp)

class PlanStep(BaseModel):

step: str = Field(

description="A single decomposition step describing part of the plan to
generate the SQL query.")

class DecompositionPlanOutput(BaseModel):

steps: listPlanStep = Field(

description="Ordered decomposition steps for the SQL generation plan.",

min_length=1,)

Listing G.2: Pydantic model for the structured output step-by-step
planning.

[⬇](data:text/plain;base64,CnsidHlwZSI6ICJqc29uX3NjaGVtYSIsCiAgICAibmFtZSI6ICJEZWNvbXBvc2l0aW9uUGxhbk91dHB1dCIsCiAgICAic3RyaWN0IjogdHJ1ZSwKICAgICJzY2hlbWEiOiB7CiAgICAgICIkZGVmcyI6IHsKICAgICAgICAiUGxhblN0ZXAiOiB7CiAgICAgICAgICAiYWRkaXRpb25hbFByb3BlcnRpZXMiOiBmYWxzZSwKICAgICAgICAgICJwcm9wZXJ0aWVzIjogewogICAgICAgICAgICAic3RlcCI6IHsKICAgICAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiQSBzaW5nbGUgZGVjb21wb3NpdGlvbiBzdGVwIGRlc2NyaWJpbmcgcGFydCBvZiB0aGUgcGxhbiB0byBnZW5lcmF0ZSB0aGUgU1FMIHF1ZXJ5LiIsCiAgICAgICAgICAgICAgInRpdGxlIjogIlN0ZXAiLAogICAgICAgICAgICAgICJ0eXBlIjogInN0cmluZyIKICAgICAgICAgICAgfQogICAgICAgICAgfSwKICAgICAgICAgICJyZXF1aXJlZCI6IFsKICAgICAgICAgICAgInN0ZXAiCiAgICAgICAgICBdLAogICAgICAgICAgInRpdGxlIjogIlBsYW5TdGVwIiwKICAgICAgICAgICJ0eXBlIjogIm9iamVjdCIKICAgICAgICB9CiAgICAgIH0sCiAgICAgICJhZGRpdGlvbmFsUHJvcGVydGllcyI6IGZhbHNlLAogICAgICAicHJvcGVydGllcyI6IHsKICAgICAgICAic3RlcHMiOiB7CiAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiT3JkZXJlZCBkZWNvbXBvc2l0aW9uIHN0ZXBzIGZvciB0aGUgU1FMIGdlbmVyYXRpb24gcGxhbi4iLAogICAgICAgICAgIml0ZW1zIjogewogICAgICAgICAgICAiJHJlZiI6ICIjLyRkZWZzL1BsYW5TdGVwIgogICAgICAgICAgfSwKICAgICAgICAgICJtaW5JdGVtcyI6IDEsCiAgICAgICAgICAidGl0bGUiOiAiU3RlcHMiLAogICAgICAgICAgInR5cGUiOiAiYXJyYXkiCiAgICAgICAgfQogICAgICB9LAogICAgICAicmVxdWlyZWQiOiBbCiAgICAgICAgInN0ZXBzIgogICAgICBdLAogICAgICAidGl0bGUiOiAiRGVjb21wb3NpdGlvblBsYW5PdXRwdXQiLAogICAgICAidHlwZSI6ICJvYmplY3QiCiAgICB9CiAgfQo=)

{"type": "json_schema",

"name": "DecompositionPlanOutput",

"strict": true,

"schema": {

"\$defs": {

"PlanStep": {

"additionalProperties": false,

"properties": {

"step": {

"description": "A single decomposition step describing part of the plan
to generate the SQL query.",

"title": "Step",

"type": "string"

}

},

"required":

"step"

,

"title": "PlanStep",

"type": "object"

}

},

"additionalProperties": false,

"properties": {

"steps": {

"description": "Ordered decomposition steps for the SQL generation
plan.",

"items": {

"\$ref": "#/\$defs/PlanStep"

},

"minItems": 1,

"title": "Steps",

"type": "array"

}

},

"required":

"steps"

,

"title": "DecompositionPlanOutput",

"type": "object"

}

}

Listing G.3: Pydantic format converted to JSON schema for API calls.
````
