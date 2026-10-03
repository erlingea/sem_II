// =========================================================================
// CLASIFICACIÓN MULTITEMPORAL Y DETECCIÓN DE CRECIMIENTO URBANO (SJL)
// =========================================================================

// 1. CONFIGURACIÓN DE AÑOS Y ASSETS
var assetsSJL = {
  '2018': 'projects/notional-mantra-472516-c7/assets/SJL_2018',
  '2019': 'projects/notional-mantra-472516-c7/assets/SJL_2019',
  '2020': 'projects/notional-mantra-472516-c7/assets/SJL_2020',
  '2021': 'projects/notional-mantra-472516-c7/assets/SJL_2021',
  '2022': 'projects/notional-mantra-472516-c7/assets/SJL_2022',
  '2023': 'projects/notional-mantra-472516-c7/assets/SJL_2023' 
};

var anioBase = '2018'; 

var visColorVerdadero = {
  bands: ['b3', 'b2', 'b1'], // Rojo, Verde, Azul
  min: 400,
  max: 3000
};

// 2. CARGAR GEOMETRÍA OFICIAL
var asset = ee.FeatureCollection('projects/notional-mantra-472516-c7/assets/Distrital_INEI');
var roi = asset.filter(ee.Filter.eq('DISTRITO', 'SAN JUAN DE LURIGANCHO')).geometry();

Map.centerObject(roi, 13);
Map.addLayer(ee.Image().paint(roi, 0, 2), {palette: ['000000']}, 'Límite Oficial SJL');

// 3. FUNCIÓN MAESTRA DE PREPROCESAMIENTO 
function prepararBandas(imagenBase) {
  var ndbi = imagenBase.normalizedDifference(['b9', 'b7']).rename('NDBI');
  var ndvi = imagenBase.normalizedDifference(['b7', 'b3']).rename('NDVI');
  var ui = imagenBase.normalizedDifference(['b10', 'b7']).rename('UI');
  var bsi = imagenBase.expression(
    '((b9 + b3) - (b7 + b1)) / ((b9 + b3) + (b7 + b1))', {
      'b1': imagenBase.select('b1'),
      'b3': imagenBase.select('b3'),
      'b7': imagenBase.select('b7'),
      'b9': imagenBase.select('b9')
  }).rename('BSI');

  var srtm = ee.Image('USGS/SRTMGL1_003').clip(roi);
  var elevacion = srtm.select('elevation').rename('Elevacion');
  var pendiente = ee.Terrain.slope(elevacion).rename('Pendiente');
  
  return imagenBase.addBands([ndbi, ndvi, ui, bsi, elevacion, pendiente]);
}

var bandasClave = ['NDBI', 'NDVI', 'UI', 'BSI', 'Elevacion', 'Pendiente'];

// Funciones de Cálculo de Área
var calcularAreaKm2 = function(imagenBinaria) {
  var area = imagenBinaria.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: roi, scale: 10, maxPixels: 1e13
  });
  return ee.Number(area.get('classification')).divide(1000000);
};

var reporteAnual = function(mapaClasificado, anio) {
  var areaUrbana = calcularAreaKm2(mapaClasificado.eq(1));
  var areaNoUrbana = calcularAreaKm2(mapaClasificado.eq(0));
  var areaTotal = areaUrbana.add(areaNoUrbana);

  print('📊 --- ESTADÍSTICAS ' + anio + ' ---');
  print('Urbano (km²):', areaUrbana);
  print('No Urbano (km²):', areaNoUrbana);
  print('Total (Urbano + No Urbano) (km²):', areaTotal);
};

// 4. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO (Solo usando año base)
if (typeof urban !== 'undefined' && typeof non_urban !== 'undefined') {
  
  var imgBase = ee.Image(assetsSJL[anioBase]).clip(roi);
  var procesadaBase = prepararBandas(imgBase);
  
  Map.addLayer(imgBase, visColorVerdadero, 'Color Real ' + anioBase, true);
  
  // -------------------------------------------------------------------------
  // PUNTOS LIMITADA ALEATORIAMENTE
  // -------------------------------------------------------------------------
  var puntosEntrenamiento = urban.merge(non_urban);
  
  var datosMuestreados = procesadaBase.select(bandasClave).sampleRegions({
    collection: puntosEntrenamiento,
    properties: ['class'],
    scale: 1,
    tileScale: 16 // <-- TRUCO 1: Sube a 16 para procesar polígonos grandes
  })
  .randomColumn('filtro_limite') 
  .filter(ee.Filter.lt('filtro_limite', 0.05)); // <-- TRUCO 2: Mantiene solo el 5% 
  // -------------------------------------------------------------------------
  
  // División 70/30 sobre el 5% de puntos que sobrevivieron al filtro
  var datosConRandom = datosMuestreados.randomColumn('random_split');
  
  var clasificador = ee.Classifier.smileRandomForest(100).train({
    features: datosConRandom.filter(ee.Filter.lt('random_split', 0.7)),
    classProperty: 'class',
    inputProperties: bandasClave
  });
  
  var mapaUrbanoBase = procesadaBase.select(bandasClave).classify(clasificador);
  Map.addLayer(mapaUrbanoBase.updateMask(mapaUrbanoBase.eq(1)), 
               {palette: ['FF0000']}, 'Mancha Urbana ' + anioBase + ' (Base)', false);

  reporteAnual(mapaUrbanoBase, anioBase);
  print('===================================================');

  // =======================================================================
  // EXPORTACIÓN DEL AÑO BASE (2018)
  // =======================================================================
  Export.image.toDrive({
    image: mapaUrbanoBase.toByte(), // Convierte booleano a entero de 8 bits
    description: 'Mancha_Urbana_SJL_' + anioBase,
    folder: 'Dataset_ConvLSTM',
    scale: 10, // Resolución de 10m para que Colab lo soporte bien en RAM
    region: roi,
    maxPixels: 1e13
  });

  // 5. PROCESAMIENTO DINÁMICO DE LOS DEMÁS AÑOS
  var listaAnios = Object.keys(assetsSJL);
  
  listaAnios.forEach(function(anio) {
    if (anio !== anioBase) { 
      try {
        var imgAnio = ee.Image(assetsSJL[anio]).clip(roi);
        var procesadaAnio = prepararBandas(imgAnio);
        
        Map.addLayer(imgAnio, visColorVerdadero, 'Color Real ' + anio, false);
        
        var mapaUrbanoAnio = procesadaAnio.select(bandasClave).classify(clasificador);
        var crecimiento = mapaUrbanoAnio.subtract(mapaUrbanoBase).eq(1);
        
        var colorHex = Math.floor(Math.random()*16777215).toString(16);
        while(colorHex.length < 6) { colorHex = "0" + colorHex; }
        
        Map.addLayer(mapaUrbanoAnio.updateMask(mapaUrbanoAnio.eq(1)), 
                     {palette: ['FFA500']}, 'Mancha Urbana ' + anio, false);
                     
        Map.addLayer(crecimiento.updateMask(crecimiento.eq(1)), 
                     {palette: [colorHex]}, 'Crecimiento ' + anioBase + ' a ' + anio, true);
        
        reporteAnual(mapaUrbanoAnio, anio);
        
        var areaCrecimiento = calcularAreaKm2(crecimiento);
        print('📈 Nuevas construcciones desde ' + anioBase + ' (km²):', areaCrecimiento);
        print('===================================================');
        
        // =======================================================================
        // EXPORTACIÓN DE LOS SIGUIENTES AÑOS AL DRIVE
        // =======================================================================
        Export.image.toDrive({
          image: mapaUrbanoAnio.toByte(),
          description: 'Mancha_Urbana_SJL_' + anio,
          folder: 'Dataset_ConvLSTM',
          scale: 10,
          region: roi,
          maxPixels: 1e13
        });
        
      } catch (e) {
        print('⚠️ No se pudo procesar el año ' + anio + '. Verifica la ruta del asset.');
      }
    }
  });
  
} else {
  print('⚠️ ACCIÓN REQUERIDA: Define tus FeatureCollections "urban" y "non_urban" en la capa base.');
}

