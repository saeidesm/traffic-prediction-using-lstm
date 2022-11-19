#Traffic Congestion Prediction Using LSTMs
* **Dataset:** City od Madrid traffic administrators deployed over 3000 traffic sensors to gather various traffic parameters including average traffic speed, density and occupancy. Aggregated data is published as an IoT service using a [RESTful API](https://informo.madrid.es/informo/tmadrid/pm.xml) and data is updated every 5 minutes. 
* Investigated the correlation between different data points in order to feed the network
* Used **stacked LSTM** to fit three days of Madrid traffic data
* Used the sliding window analogy to predict next four steps of speed, density and occupancy 
* Analyzed the predicted data to make decision upon the congestion occurence
* Results demonstrated 99% accuracy in parameters prediction
<br>_Keywords(Time Series Data, LSTM, Traffic Prediction)_
