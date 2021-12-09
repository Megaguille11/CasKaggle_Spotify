# Cas Kaggle: Spotify Music DataBase
## Repositori corresponent al cas Kaggle aplicat sobre la base de dades de Spotify
### Autor: Guillem Centelles Pavon

En aquest repositori es troba la feina realitzada per tal de crear un model classificador capaç de prediure si una cançó és acústica a partir d'atributs tècnics d'aquesta.

_(Nota: el fitxer .csv de la BBDD es trova comprimit per tal de reduïrne el seu pes, descomprimir en el directori /data abans d'executar qualsevol part del codi)_

## Introducció

Com ja s'ha especificat en el títol, la base de dades seleccionada per a aquest projecte és la corresponent a les cançons que es troben dins la plataforma de música en streaming Spotify, la qual per a cada entrada, és a dir, per a cada cançó, té guardats els següents atributs:

* _genre_: Gènere musical al qual pertany la cançó.
* _atrist_name_: Nom de artista autor de la cançó.
* _track_name_: Nom de la cançó.
* _track_id_: Identificador únic assignat a una cançó en concret.
* _popularity_: Popilaritat de la cançó entre els usuaris que l'han escoltada (Enter en el rang (0, 100)).
* _acousticness_: Mesura de confiança sobre si la cançó és acústica o no (Decimal en el rang (0, 1)).
* _danceability_: Mesura de com de bona és la cançó per a ser ballada, basat en elements musicals de la propia (Decimal en el rang (0, 1)).
* _duration_ms_: Duració de la cançó, expresada en milisegons.
* _energy_: Mesura que representa la percepció general sobre la intensitat i activitat d'una cançó, amb l'exemple d'una cançó de heavy metal puntuant alt en aquesta categoria i un preludi de Bachuntuant baix (Decimal en el rang (0, 1)).
* _instrumentalness_: Probabilitat que la cançó no sigui cantada (tractant expressions com "Ooh" i "aah" com a part instrumental. Decimal en el rang (0, 1)).
* _key_: Clau musical en la que la cançó ha estat composta.
* _liveness_: Probabilitat de que la cançó sigui una grabació del artista tocan amb públic (Decimal en el rang (0, 1)).
* _loudness_: La intensitat sonora mitjana d'una cançó mesurada en decibels (Normalment en el rang (-60, 0)).
* _mode_: Modalitat de la cançó (Major o Menor).
* _speechiness_: Presència de paraules parlades en una cançó, sent cançons amb valors contigut com podcasts o poesia recitada (Decimal en el rang (0, 1)).
* _tempo_: La estimació del tempo mitjà d'una cançó, mesurat en BPM (pulsacions per minut).
* _time_signature_: Estimació del compàs de la cançó, és a dir, en número de pulsacions d'aquesta per cada unitat de mesura.
* _valence_: Mesura del "positivisme" de la cançó, a major valor, més alegre i eufòrica és la cançó (Decimal en el rang (0, 1)). 

En el notre cas, establirem com a variable objectiu l'anomenada acousticness, sobre la qual crearem un model classificador que sigui capaç de decidir si una cançó és acústica o no.
