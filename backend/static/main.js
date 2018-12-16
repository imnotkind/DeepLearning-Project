
let refresh = function(){
    let cfg = {
        pieceTheme: 'static/chessboardjs-0.3.0/img/chesspieces/wikipedia/{piece}.png'
    }

    fetch("http://141.223.163.184:5000/getinfo")
    .then(res => res.json())
    .then(function(data){
        console.log(data)

        $("#player_ann").html(data["player"]["ann"])
        $("#player_move").html(data["player"]["move"])
    
        if(data["player"]["fen"] != "None")
            cfg["position"] = data["player"]["fen"]
        
        let board1 = ChessBoard('board1', cfg)
        $(window).resize(board1.resize);


        $("#computer_ann").html(data["computer"]["ann"])
        $("#computer_move").html(data["computer"]["move"])

        if(data["computer"]["fen"] != "None")
            cfg["position"] = data["computer"]["fen"]
        
        let board2 = ChessBoard('board2', cfg)
        $(window).resize(board2.resize);


    })

    


}

setInterval(refresh, 1000)