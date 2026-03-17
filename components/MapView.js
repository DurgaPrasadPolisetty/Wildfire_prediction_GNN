import {
  MapContainer,
  TileLayer,
  Marker,
  useMapEvents,
  useMap
} from "react-leaflet";

import { useEffect, useState } from "react";
import axios from "axios";
import L from "leaflet";

import "leaflet.heat";

import FireAnimation from "./FireAnimation";
import { predictFire } from "../services/api";

import fireIconImg from "../assets/fire.png";

const fireIcon = new L.Icon({
  iconUrl: fireIconImg,
  iconSize: [32,32],
  iconAnchor: [16,32]
});

/* CLICK HANDLER */

function MapClickHandler({setMarker,setPrediction}){

  useMapEvents({

    async click(e){

      const lat = e.latlng.lat;
      const lon = e.latlng.lng;

      setMarker([lat,lon]);

      const result = await predictFire(lat,lon);

      setPrediction(result);

    }

  });

  return null;
}

/* HEATMAP LAYER */

function HeatmapLayer({points}){

  const map = useMap();

  useEffect(()=>{

    if(points.length === 0) return;

    const heat = L.heatLayer(points,{
      radius:25,
      blur:15,
      maxZoom:10
    });

    heat.addTo(map);

    return ()=> map.removeLayer(heat);

  },[points,map]);

  return null;
}

export default function MapView({setPrediction}){

  const [marker,setMarker] = useState(null);
  const [fires,setFires] = useState([]);
  const [heatPoints,setHeatPoints] = useState([]);

  useEffect(()=>{

    async function loadFires(){

      try{

        const url =
        "https://firms.modaps.eosdis.nasa.gov/api/area/csv/dfa2c33100e185c14f840fc2b0d2184d/VIIRS_SNPP_NRT/world/1";

        const res = await axios.get(url);

        const rows = res.data.split("\n").slice(1);

        const firePoints = rows.map(r=>{

          const c=r.split(",");

          return{
            lat:parseFloat(c[0]),
            lon:parseFloat(c[1])
          };

        })
        .filter(p=>
          !isNaN(p.lat) &&
          p.lat>6 &&
          p.lat<38 &&
          p.lon>68 &&
          p.lon<98
        );

        setFires(firePoints);

        const heat = firePoints.map(p=>[
          p.lat,
          p.lon,
          1
        ]);

        setHeatPoints(heat);

      }catch(e){

        console.log("NASA API error",e);

      }

    }

    loadFires();

  },[]);

  return(

<MapContainer
center={[22.97,78.65]}
zoom={5}
style={{height:"100%",width:"100%"}}
>

<TileLayer
url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
/>

<MapClickHandler
setMarker={setMarker}
setPrediction={setPrediction}
/>

<HeatmapLayer points={heatPoints}/>

{fires.map((f,i)=>(

<Marker
key={i}
position={[f.lat,f.lon]}
icon={fireIcon}
/>

))}

<FireAnimation fires={fires}/>

{marker &&

<Marker
position={marker}
icon={fireIcon}
/>

}

</MapContainer>

  );

}