SCENE = TowerOfHanoi
SOURCE = tower_of_hanoi.py

.PHONY: render preview clean

render:
	uv run manim -qh $(SOURCE) $(SCENE)

preview:
	uv run manim -pql $(SOURCE) $(SCENE)

clean:
	rm -rf media
