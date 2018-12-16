
let refresh = function(){
    let cfg = {
        pieceTheme: 'static/chessboardjs-0.3.0/img/chesspieces/wikipedia/{piece}.png'
    }
    let fen_player = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let fen_computer = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let ann_player = "PLAYER"
    let ann_computer = "COMPUTER"

    fetch("http://141.223.163.184:5000/getinfo")
    .then(res => res.json())
    .then(function(data){
        console.log(data)

        $("#player_ann").html(data["player"]["ann"])
    
        if(data["player"]["fen"] != "None")
            cfg["position"] = data["player"]["fen"]
        
        ChessBoard('board1', cfg)


        $("#computer_ann").html(data["computer"]["ann"])

        if(data["computer"]["fen"] != "None")
            cfg["position"] = data["computer"]["fen"]
        
        ChessBoard('board2', cfg)


    })

    


}

setInterval(refresh, 1000)