# Chess AI & Annotater



## Structure

체스를 하면서 각 행동에 대한 annotation을 딥러닝 모델로 구하고 표시

---



수동모드일때 

- 플레이어에게 다음 수 query
- player response
  - MOVE : 그 한 수의 vaildation 체크
  - START :  

---

1. end

   플레이어 수 + 컴퓨터 수로 게임이 끝난 경우에 is_end 전달 : "win", "lose"

2. continue

   플레이어 수의 validation 체크로 is_valid 주고, 그것에 따라 NUGU에서 말을 해줌

   valid하지 않은 수에 대해서는 체스판에 적용하지 않음

## Actions

#### action.game.start(START)

- player_color(invalid, black, white) -> FILTER_COLOR
- computer_move

#### action.input.move(MOVE)

- **game.end** : is_end_game(win,lose) -> FILTER_END_GAME
- **game.continued** : is_valid_move(true, false) -> FILTER_VALID_MOVE





## Learning

