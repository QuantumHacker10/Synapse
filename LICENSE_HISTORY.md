# Historique des licences — Synapse OMNIA

Ce document clarifie l'évolution des licences du projet pour les contributeurs,
utilisateurs et auditeurs.

## Chronologie

| Date | Commit | Licence | Notes |
|---|---|---|---|
| 2026-07-18 | `3a76f01` | **MIT** | Import initial SYNAPSE OMNIA 1.1 |
| 2026-07-19 | `22321d8` | **Propriétaire** | Remplacement temporaire par une licence anti-plagiat propriétaire |
| 2026-07-20 | `3f08df1` | **MIT** | Retour à l'open source MIT (release v1.2.0) |

## Statut actuel

**Licence MIT** — le projet est open source depuis la release **v1.2.0** (2026-07-20).

Voir [`LICENSE`](LICENSE) pour le texte complet.

## Implications pour les contributeurs

- Toutes les contributions acceptées et mergées à partir du **2026-07-20** sont
  sous licence MIT.
- Le code publié entre le **2026-07-19** et le **2026-07-20** sous licence propriétaire
  a été **re-licencié en MIT** lors du commit `3f08df1`. Les tags `v1.1.0` et antérieurs
  reflètent cette période ; le tag `v1.2.0` marque le retour définitif à MIT.
- En contribuant, vous acceptez que vos contributions soient intégrées sous la
  licence MIT du projet (voir [CONTRIBUTING.md](CONTRIBUTING.md)).

## Implications pour les utilisateurs

- Vous pouvez utiliser, modifier, distribuer et créer des œuvres dérivées du code
  source sous les conditions de la licence MIT.
- Les marques **SYNAPSE OMNIA**, **Synapse Engine** et **Synapse Studio** ne sont
  pas couvertes par la licence logicielle (voir [`COPYRIGHT`](COPYRIGHT)).
- Les dépendances tierces ont leurs propres licences — voir
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Fichiers légaux associés

| Fichier | Rôle |
|---|---|
| [`LICENSE`](LICENSE) | Texte MIT actuel |
| [`COPYRIGHT`](COPYRIGHT) | Avis copyright et marques |
| [`NOTICE`](NOTICE) | Avis standard (crédits, marques) |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Licences des dépendances |

## Vérification git

Pour consulter l'historique des changements de licence :

```bash
git log --oneline -- LICENSE
```

Résultat attendu :

```
3f08df1 chore: passer à la licence MIT (open source)
22321d8 legal: replace MIT with proprietary anti-plagiarism license
3a76f01 Import SYNAPSE OMNIA 1.1: moteur, G-DNN Studio, site, README et licence.
```

## Questions

Pour toute question sur la licence ou l'historique, ouvrez une
[issue](https://github.com/QuantumHacker10/Synapse/issues) avec le label `legal`.
