
let refresh = function(){
    let cfg = {
        pieceTheme: 'static/chessboardjs-0.3.0/img/chesspieces/wikipedia/{piece}.png'
    }
    let fen_player = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let fen_computer = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    let ann_player = "FUCK"
    let ann_computer = "YOU"

    fetch("http://141.223.163.184:5000/getinfo")
    .then(res => res.json())
    .then(function(data){
        console.log(data)
        fen_player = data["fen_player"]
        fen_computer = data["fen_computer"]
        ann_player = data["ann_player"]
        ann_computer = data["ann_computer"]


        $("#player_ann").html(ann_player)
    
        cfg["position"] = fen_player
        ChessBoard('board1', cfg)

        console.log(cfg)

        $("#computer_ann").html(ann_computer)

        cfg["position"] = fen_computer
        ChessBoard('board2', cfg)

        console.log(cfg)

    })

    


}

setInterval(refresh, 1000)