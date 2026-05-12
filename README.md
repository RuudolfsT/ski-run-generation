
# Maģistra darba **AUTOMATIZĒTA KALNU SLĒPOŠANAS TRAŠU ĢENERĒŠANA, IZMANTOJOT DIGITĀLOS AUGSTUMA MODEĻUS UN OPTIMIZĀCIJAS METODES** pirmkoda repozitorijs

Repozitorijs ietver 2 failus, kas izstrādāti darba ietvaros: 
## *start_point_selection_model.model3*
 * Satur izstrādāto modeli reljefa datu apstrādei un potenciālo slēpošanas trašu sākumpunktu atlasei (aprakstīts 5. nodaļā) kādam konkrētam digitālajam augstuma modelim, kuru var ielādēt, piemēram, no https://opentopography.org/
 * Modeļa izmantošanai nepieciešama lejupielādēta **QGIS** programmatūra ar **QuickOSM** spraudni:
    * https://qgis.org/
    * https://plugins.qgis.org/plugins/QuickOSM/
 * Failu var atvērt kā **QGIS Model Designer** modeli, kur tam no sākuma var mainīt  parametrus, kas aprakstīti darbā
 * Svarīgi norādīt konkrētu _output_folder_, kurā tiks saglabāti modeļa izvades faili (tie tiks nolasīti **Python** kodā)
## 

## 
* Satur izstrādāto **Python** kodu slēpošanas trašu un koridoru ģenerēšanai (aprakstīts 6. nodaļā), izmantojot modeļa izgūtos reljefa datus un atlasītos sākumpunktus
* Svarīgi, lai pirms koda darbināšanas tiktu izpildīts iepriekš minētais modelis
* Visiem modeļa izvades failiem jāatrodas zināmā mapē (pēc noklusējuma `data/` mape), lai programma varētu tos nolasīt
* Koda sākumā atrodas vairāki parametru mainīgie (ar piešķirtām noklusējuma vērtībām), kurus mainot, var iegūt dažādus ģenerēšanas rezultātus, piemēram, palielināt ģenerēto trašu skaitu katram punktam (parametri ir aprakstīti darbā)
* Jāpārliecinās, ka ir lejupielādētas vajadzīgās bibliotēkas: 
    * `pip install rasterio geopandas numpy matplotlib shapely`
* Darbināt var ar: 
    * `python main.py` 
* Programma atgriež trīs failus, kurus pēc tam var atvērt **QGIS**:
    * ***generated_ski_runs.gpkg*** - satur 4 vektordatu slāņus ar klasificētām slēpošanas trasēm (visas trases, zilās trases, sarkanās trases un melnās trases) un to atribūtiem
    * ***generated_corridors.tif*** - satur rastra datus par katram sākumpunktam ģenerēto koridoru
    * ***cost_raster.tif*** - satur izveidoto izmaksu virsmas rastru
## 

*Autors: Rūdolfs Arvīds Truls, rt21028*
