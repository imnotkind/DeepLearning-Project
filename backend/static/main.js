


let refresh = function(){
    let cfg1 = {
        position: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R',
        pieceTheme: 'static/chessboardjs-0.3.0/img/chesspieces/wikipedia/{piece}.png'
    }
    ChessBoard('board1', cfg1);

    let cfg2 = {
        position: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R',
        pieceTheme: 'static/chessboardjs-0.3.0/img/chesspieces/wikipedia/{piece}.png'
    }
    ChessBoard('board2', cfg2);
}

setInterval(refresh, 1000);