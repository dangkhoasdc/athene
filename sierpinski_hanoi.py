"""
Tower of Hanoi State Graph as Sierpiński Triangle.

The state graph of the Tower of Hanoi puzzle forms a Sierpiński triangle:
- Each NODE represents a configuration (which disks are on which pegs)
- Each EDGE represents a valid move between two configurations
- For n disks, there are 3^n nodes (states)
- The three corners represent all disks on a single peg

The recursive structure:
- n=1: Simple triangle (3 nodes, 3 edges)
- n=2: Three triangles connected at corners (9 nodes)
- n=k: Take n=k-1 graph, make 3 copies, connect at corners
"""

from manim import *
import itertools


class SierpinskiHanoi(Scene):
    """Visualize Tower of Hanoi state graph as Sierpiński triangle."""

    NUM_DISKS = 3
    ANIMATION_SPEED = 0.6

    def construct(self) -> None:
        """Build the visualization."""
        # Title
        title = Text("Tower of Hanoi State Graph", font_size=36)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        subtitle = Text(
            f"n = {self.NUM_DISKS} disks → {3**self.NUM_DISKS} states",
            font_size=24,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(subtitle))

        # Generate all states and build the graph
        states = self._generate_all_states()
        positions = self._compute_sierpinski_positions(states)
        edges = self._generate_edges(states)

        # Draw the graph
        graph_group = self._draw_graph(states, positions, edges)

        self.wait(1)

        # Animate the solution path
        self._animate_solution(states, positions)

        self.wait(2)

    def _generate_all_states(self) -> list[tuple[tuple[int, ...], ...]]:
        """Generate all 3^n possible states.
        
        A state is a tuple of 3 tuples, each representing disks on a peg.
        Disks are numbered 1 (smallest) to n (largest).
        Each tuple is sorted with smallest on top (at index 0).
        """
        n = self.NUM_DISKS
        states = []

        # Each disk can be on peg 0, 1, or 2
        for assignment in itertools.product(range(3), repeat=n):
            # assignment[i] = which peg disk (i+1) is on
            pegs: list[list[int]] = [[], [], []]
            for disk, peg in enumerate(assignment, start=1):
                pegs[peg].append(disk)
            # Sort each peg (smallest disk on top)
            state = tuple(tuple(sorted(p)) for p in pegs)
            states.append(state)

        return states

    def _compute_sierpinski_positions(
        self, states: list[tuple]
    ) -> dict[tuple, np.ndarray]:
        """Compute position of each state in Sierpiński triangle layout.
        
        The position is computed using barycentric coordinates based on
        where the largest disk is, then recursively for smaller disks.
        """
        # Main triangle vertices (corners where all disks are on one peg)
        size = 5.0
        center = DOWN * 0.5

        # Vertices: top = peg 0, bottom-left = peg 1, bottom-right = peg 2
        vertices = [
            center + UP * size * 0.5,                          # peg 0 (top)
            center + DOWN * size * 0.25 + LEFT * size * 0.45,  # peg 1 (bottom-left)
            center + DOWN * size * 0.25 + RIGHT * size * 0.45, # peg 2 (bottom-right)
        ]

        positions = {}
        for state in states:
            pos = self._state_to_position(state, vertices, self.NUM_DISKS)
            positions[state] = pos

        return positions

    def _state_to_position(
        self,
        state: tuple[tuple[int, ...], ...],
        vertices: list[np.ndarray],
        depth: int,
    ) -> np.ndarray:
        """Recursively compute position of a state.
        
        For the largest disk at this level, determine which sub-triangle
        we're in, then recurse for smaller disks.
        """
        if depth == 0:
            # Base case: center of current triangle
            return sum(vertices) / 3

        # Find which peg has disk 'depth' (the largest at this level)
        largest_disk = depth
        peg_of_largest = -1
        for peg_idx, peg in enumerate(state):
            if largest_disk in peg:
                peg_of_largest = peg_idx
                break

        # The sub-triangle for this peg
        # Compute midpoints
        mid01 = (vertices[0] + vertices[1]) / 2
        mid02 = (vertices[0] + vertices[2]) / 2
        mid12 = (vertices[1] + vertices[2]) / 2

        # Sub-triangles (each is half the size, at a corner)
        sub_triangles = [
            [vertices[0], mid01, mid02],  # peg 0: top sub-triangle
            [mid01, vertices[1], mid12],  # peg 1: bottom-left sub-triangle
            [mid02, mid12, vertices[2]],  # peg 2: bottom-right sub-triangle
        ]

        return self._state_to_position(
            state, sub_triangles[peg_of_largest], depth - 1
        )

    def _generate_edges(
        self, states: list[tuple]
    ) -> list[tuple[tuple, tuple, int]]:
        """Generate all valid edges (moves) between states.
        
        Returns list of (state1, state2, disk_moved).
        """
        state_set = set(states)
        edges = []
        seen = set()

        for state in states:
            pegs = [list(p) for p in state]

            for from_peg in range(3):
                if not pegs[from_peg]:
                    continue
                # Top disk on this peg (smallest number = smallest disk)
                disk = pegs[from_peg][0]

                for to_peg in range(3):
                    if from_peg == to_peg:
                        continue
                    # Check if move is valid (can't place on smaller disk)
                    if pegs[to_peg] and pegs[to_peg][0] < disk:
                        continue

                    # Create new state
                    new_pegs = [list(p) for p in pegs]
                    new_pegs[from_peg] = new_pegs[from_peg][1:]
                    new_pegs[to_peg] = [disk] + new_pegs[to_peg]
                    new_state = tuple(tuple(p) for p in new_pegs)

                    if new_state in state_set:
                        edge_key = tuple(sorted([state, new_state]))
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append((state, new_state, disk))

        return edges

    def _draw_graph(
        self,
        states: list[tuple],
        positions: dict[tuple, np.ndarray],
        edges: list[tuple[tuple, tuple, int]],
    ) -> VGroup:
        """Draw the state graph."""
        graph_group = VGroup()

        # Color edges by which disk moves
        disk_colors = [RED, GREEN, BLUE, YELLOW, PURPLE, ORANGE]

        # Draw edges
        self.edge_lines = {}
        for state1, state2, disk in edges:
            color = disk_colors[(disk - 1) % len(disk_colors)]
            line = Line(
                positions[state1],
                positions[state2],
                stroke_width=1.5,
                color=color,
                stroke_opacity=0.6,
            )
            self.edge_lines[(state1, state2)] = line
            self.edge_lines[(state2, state1)] = line
            graph_group.add(line)

        self.play(Create(graph_group), run_time=1.5)

        # Draw nodes
        self.nodes = {}
        node_group = VGroup()
        
        # Find corner states (all disks on one peg)
        all_on_peg_0 = (tuple(range(1, self.NUM_DISKS + 1)), (), ())
        all_on_peg_1 = ((), tuple(range(1, self.NUM_DISKS + 1)), ())
        all_on_peg_2 = ((), (), tuple(range(1, self.NUM_DISKS + 1)))
        corner_states = {all_on_peg_0, all_on_peg_1, all_on_peg_2}

        for state in states:
            pos = positions[state]
            if state in corner_states:
                # Corner nodes (start/end positions) are larger
                if state == all_on_peg_0:
                    color = GREEN  # Start
                    radius = 0.12
                elif state == all_on_peg_2:
                    color = BLUE  # End (destination)
                    radius = 0.12
                else:
                    color = YELLOW
                    radius = 0.1
            else:
                color = WHITE
                radius = 0.05

            node = Dot(pos, radius=radius, color=color)
            self.nodes[state] = node
            node_group.add(node)

        self.play(FadeIn(node_group), run_time=0.5)

        # Add corner labels
        labels = VGroup()
        label_texts = ["All on A\n(Start)", "All on B", "All on C\n(Goal)"]
        label_states = [all_on_peg_0, all_on_peg_1, all_on_peg_2]
        label_directions = [UP, DOWN + LEFT, DOWN + RIGHT]
        
        for text, state, direction in zip(label_texts, label_states, label_directions):
            label = Text(text, font_size=14, color=GRAY)
            label.next_to(positions[state], direction, buff=0.15)
            labels.add(label)

        self.play(FadeIn(labels), run_time=0.5)

        # Legend
        legend = VGroup()
        legend_title = Text("Edge color = disk moved:", font_size=14)
        legend.add(legend_title)
        
        for i in range(self.NUM_DISKS):
            color = disk_colors[i % len(disk_colors)]
            disk_indicator = VGroup(
                Line(ORIGIN, RIGHT * 0.3, stroke_width=3, color=color),
                Text(f"Disk {i + 1}", font_size=12, color=color),
            ).arrange(RIGHT, buff=0.1)
            legend.add(disk_indicator)

        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(DR, buff=0.3)
        self.play(FadeIn(legend), run_time=0.5)

        graph_group.add(node_group, labels, legend)
        return graph_group

    def _animate_solution(
        self, states: list[tuple], positions: dict[tuple, np.ndarray]
    ) -> None:
        """Animate the optimal solution path through the graph."""
        # Generate the solution path
        solution_path = self._solve_hanoi()

        # Create a marker for current state
        start_state = (tuple(range(1, self.NUM_DISKS + 1)), (), ())
        marker = Circle(
            radius=0.18, color=YELLOW, stroke_width=3, fill_opacity=0
        )
        marker.move_to(positions[start_state])
        self.play(Create(marker))

        # Create move counter
        move_text = Text("Move: 0", font_size=20, color=YELLOW)
        move_text.to_corner(UL, buff=0.5)
        self.add(move_text)

        current_state = start_state

        for move_num, (from_peg, to_peg) in enumerate(solution_path, start=1):
            # Compute new state
            pegs = [list(p) for p in current_state]
            disk = pegs[from_peg][0]
            pegs[from_peg] = pegs[from_peg][1:]
            pegs[to_peg] = [disk] + pegs[to_peg]
            new_state = tuple(tuple(p) for p in pegs)

            # Highlight the edge being traversed
            edge_line = self.edge_lines.get((current_state, new_state))
            if edge_line:
                self.play(
                    edge_line.animate.set_stroke(opacity=1, width=4),
                    marker.animate.move_to(positions[new_state]),
                    run_time=self.ANIMATION_SPEED,
                )

            # Update move counter
            self.remove(move_text)
            move_text = Text(f"Move: {move_num}", font_size=20, color=YELLOW)
            move_text.to_corner(UL, buff=0.5)
            self.add(move_text)

            current_state = new_state

        # Final marker highlight
        self.play(
            marker.animate.set_color(GREEN).set_stroke(width=5),
            run_time=0.5,
        )

    def _solve_hanoi(self) -> list[tuple[int, int]]:
        """Generate the optimal solution as a list of (from_peg, to_peg) moves."""
        moves = []

        def hanoi(n: int, source: int, target: int, auxiliary: int) -> None:
            if n == 0:
                return
            hanoi(n - 1, source, auxiliary, target)
            moves.append((source, target))
            hanoi(n - 1, auxiliary, target, source)

        hanoi(self.NUM_DISKS, 0, 2, 1)
        return moves


class SierpinskiHanoiBuildup(Scene):
    """Show how the Sierpiński structure emerges as we add disks."""

    def construct(self) -> None:
        title = Text("Building the State Graph", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        for n in range(1, 4):
            self._show_level(n)

        self.wait(2)

    def _show_level(self, n: int) -> None:
        """Show the state graph for n disks."""
        subtitle = Text(
            f"n = {n} disk{'s' if n > 1 else ''}: {3**n} states, {(3**n - 1) * 3 // 2} edges",
            font_size=24,
            color=GRAY,
        )
        subtitle.next_to(
            self.mobjects[0] if self.mobjects else ORIGIN, DOWN, buff=0.2
        )
        self.play(FadeIn(subtitle))

        # Generate states and positions
        states = self._generate_all_states(n)
        positions = self._compute_positions(states, n)
        edges = self._generate_edges(states)

        # Draw
        graph = VGroup()
        disk_colors = [RED, GREEN, BLUE]

        # Edges
        for s1, s2, disk in edges:
            color = disk_colors[(disk - 1) % len(disk_colors)]
            line = Line(
                positions[s1], positions[s2],
                stroke_width=2, color=color, stroke_opacity=0.7
            )
            graph.add(line)

        # Nodes
        for state in states:
            radius = 0.08 if n < 3 else 0.04
            dot = Dot(positions[state], radius=radius, color=WHITE)
            graph.add(dot)

        graph.move_to(ORIGIN + DOWN * 0.5)
        self.play(Create(graph), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(graph), FadeOut(subtitle), run_time=0.5)

    def _generate_all_states(self, n: int) -> list[tuple]:
        states = []
        for assignment in itertools.product(range(3), repeat=n):
            pegs: list[list[int]] = [[], [], []]
            for disk, peg in enumerate(assignment, start=1):
                pegs[peg].append(disk)
            state = tuple(tuple(sorted(p)) for p in pegs)
            states.append(state)
        return states

    def _compute_positions(
        self, states: list[tuple], n: int
    ) -> dict[tuple, np.ndarray]:
        size = 4.5
        center = ORIGIN
        vertices = [
            center + UP * size * 0.5,
            center + DOWN * size * 0.25 + LEFT * size * 0.45,
            center + DOWN * size * 0.25 + RIGHT * size * 0.45,
        ]

        positions = {}
        for state in states:
            pos = self._state_to_position(state, vertices, n)
            positions[state] = pos
        return positions

    def _state_to_position(
        self, state: tuple, vertices: list[np.ndarray], depth: int
    ) -> np.ndarray:
        if depth == 0:
            return sum(vertices) / 3

        largest_disk = depth
        peg_of_largest = -1
        for peg_idx, peg in enumerate(state):
            if largest_disk in peg:
                peg_of_largest = peg_idx
                break

        mid01 = (vertices[0] + vertices[1]) / 2
        mid02 = (vertices[0] + vertices[2]) / 2
        mid12 = (vertices[1] + vertices[2]) / 2

        sub_triangles = [
            [vertices[0], mid01, mid02],
            [mid01, vertices[1], mid12],
            [mid02, mid12, vertices[2]],
        ]

        return self._state_to_position(
            state, sub_triangles[peg_of_largest], depth - 1
        )

    def _generate_edges(self, states: list[tuple]) -> list[tuple]:
        state_set = set(states)
        edges = []
        seen = set()

        for state in states:
            pegs = [list(p) for p in state]
            for from_peg in range(3):
                if not pegs[from_peg]:
                    continue
                disk = pegs[from_peg][0]
                for to_peg in range(3):
                    if from_peg == to_peg:
                        continue
                    if pegs[to_peg] and pegs[to_peg][0] < disk:
                        continue
                    new_pegs = [list(p) for p in pegs]
                    new_pegs[from_peg] = new_pegs[from_peg][1:]
                    new_pegs[to_peg] = [disk] + new_pegs[to_peg]
                    new_state = tuple(tuple(p) for p in new_pegs)
                    if new_state in state_set:
                        edge_key = tuple(sorted([state, new_state]))
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append((state, new_state, disk))
        return edges


class SierpinskiHanoiWithTower(Scene):
    """Side-by-side visualization: Tower of Hanoi + State Graph."""

    NUM_DISKS = 3
    ANIMATION_SPEED = 0.5

    # Tower layout
    TOWER_CENTER = LEFT * 4
    PEG_HEIGHT = 2.0
    PEG_WIDTH = 0.08
    BASE_HEIGHT = 0.1
    BASE_WIDTH = 1.3
    DISK_HEIGHT = 0.28
    MIN_DISK_WIDTH = 0.4
    MAX_DISK_WIDTH = 1.1
    PEG_SPACING = 1.5
    LIFT_HEIGHT = 1.4

    # Graph layout
    GRAPH_CENTER = RIGHT * 2.5
    GRAPH_SIZE = 4.0

    DISK_COLORS = [RED, GREEN, BLUE]

    def construct(self) -> None:
        # Title
        title = Text("Tower of Hanoi ↔ Sierpiński Graph", font_size=32)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Initialize tower
        self.peg_stacks: list[list[VMobject]] = [[], [], []]
        self._create_tower()

        # Initialize graph
        self.states = self._generate_all_states()
        self.positions = self._compute_positions()
        self.edges = self._generate_edges()
        self._create_graph()

        self.wait(1)

        # Solve with synchronized animation
        self._animated_solve()

        # Done
        complete = Text("Complete!", font_size=32, color=GREEN)
        complete.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(complete, scale=1.2))
        self.wait(2)

    def _create_tower(self) -> None:
        """Create the tower visualization."""
        tower_label = Text("Physical Puzzle", font_size=18)
        tower_label.move_to(self.TOWER_CENTER + UP * 2.3)
        self.add(tower_label)

        peg_labels = ["A", "B", "C"]
        for i, x_offset in enumerate([-self.PEG_SPACING, 0, self.PEG_SPACING]):
            x_pos = self.TOWER_CENTER[0] + x_offset
            y_base = self.TOWER_CENTER[1] - 1.0

            peg = Rectangle(
                height=self.PEG_HEIGHT, width=self.PEG_WIDTH,
                fill_color=GRAY_BROWN, fill_opacity=1, stroke_width=0,
            )
            peg.move_to([x_pos, y_base + self.PEG_HEIGHT / 2, 0])

            base = Rectangle(
                height=self.BASE_HEIGHT, width=self.BASE_WIDTH,
                fill_color=GRAY_BROWN, fill_opacity=1, stroke_width=0,
            )
            base.move_to([x_pos, y_base, 0])

            label = Text(peg_labels[i], font_size=16, color=GRAY)
            label.next_to(base, DOWN, buff=0.1)

            self.add(peg, base, label)

        # Create disks
        for i in range(self.NUM_DISKS):
            width = self.MAX_DISK_WIDTH - (
                self.MAX_DISK_WIDTH - self.MIN_DISK_WIDTH
            ) * i / max(1, self.NUM_DISKS - 1)
            disk = RoundedRectangle(
                height=self.DISK_HEIGHT, width=width, corner_radius=0.05,
                fill_color=self.DISK_COLORS[i], fill_opacity=1,
                stroke_color=WHITE, stroke_width=1,
            )
            y_pos = self._get_disk_y(0)
            x_pos = self.TOWER_CENTER[0] - self.PEG_SPACING
            disk.move_to([x_pos, y_pos, 0])
            self.peg_stacks[0].append(disk)
            self.add(disk)

    def _create_graph(self) -> None:
        """Create the state graph visualization."""
        graph_label = Text("State Graph", font_size=18)
        graph_label.move_to(self.GRAPH_CENTER + UP * 2.3)
        self.add(graph_label)

        # Draw edges
        self.edge_lines = {}
        for s1, s2, disk in self.edges:
            color = self.DISK_COLORS[(disk - 1) % len(self.DISK_COLORS)]
            line = Line(
                self.positions[s1], self.positions[s2],
                stroke_width=1.5, color=color, stroke_opacity=0.4,
            )
            self.edge_lines[(s1, s2)] = line
            self.edge_lines[(s2, s1)] = line
            self.add(line)

        # Draw nodes
        self.node_dots = {}
        start_state = (tuple(range(1, self.NUM_DISKS + 1)), (), ())
        end_state = ((), (), tuple(range(1, self.NUM_DISKS + 1)))

        for state in self.states:
            if state == start_state:
                color, radius = GREEN, 0.1
            elif state == end_state:
                color, radius = BLUE, 0.1
            else:
                color, radius = WHITE, 0.04
            dot = Dot(self.positions[state], radius=radius, color=color)
            self.node_dots[state] = dot
            self.add(dot)

        # Current state marker
        self.current_state = start_state
        self.marker = Circle(
            radius=0.15, color=YELLOW, stroke_width=3, fill_opacity=0
        )
        self.marker.move_to(self.positions[start_state])
        self.add(self.marker)

        # Move counter
        self.move_count = 0
        self.move_text = Text("Move: 0", font_size=18, color=YELLOW)
        self.move_text.to_corner(DL, buff=0.4)
        self.add(self.move_text)

    def _generate_all_states(self) -> list[tuple]:
        states = []
        for assignment in itertools.product(range(3), repeat=self.NUM_DISKS):
            pegs: list[list[int]] = [[], [], []]
            for disk, peg in enumerate(assignment, start=1):
                pegs[peg].append(disk)
            state = tuple(tuple(sorted(p)) for p in pegs)
            states.append(state)
        return states

    def _compute_positions(self) -> dict[tuple, np.ndarray]:
        size = self.GRAPH_SIZE
        center = self.GRAPH_CENTER + DOWN * 0.3
        vertices = [
            center + UP * size * 0.45,
            center + DOWN * size * 0.22 + LEFT * size * 0.4,
            center + DOWN * size * 0.22 + RIGHT * size * 0.4,
        ]
        positions = {}
        for state in self.states:
            positions[state] = self._state_to_position(state, vertices, self.NUM_DISKS)
        return positions

    def _state_to_position(
        self, state: tuple, vertices: list[np.ndarray], depth: int
    ) -> np.ndarray:
        if depth == 0:
            return sum(vertices) / 3
        largest_disk = depth
        peg_of_largest = next(
            i for i, peg in enumerate(state) if largest_disk in peg
        )
        mid01 = (vertices[0] + vertices[1]) / 2
        mid02 = (vertices[0] + vertices[2]) / 2
        mid12 = (vertices[1] + vertices[2]) / 2
        sub_triangles = [
            [vertices[0], mid01, mid02],
            [mid01, vertices[1], mid12],
            [mid02, mid12, vertices[2]],
        ]
        return self._state_to_position(state, sub_triangles[peg_of_largest], depth - 1)

    def _generate_edges(self) -> list[tuple]:
        state_set = set(self.states)
        edges = []
        seen = set()
        for state in self.states:
            pegs = [list(p) for p in state]
            for from_peg in range(3):
                if not pegs[from_peg]:
                    continue
                disk = pegs[from_peg][0]
                for to_peg in range(3):
                    if from_peg == to_peg:
                        continue
                    if pegs[to_peg] and pegs[to_peg][0] < disk:
                        continue
                    new_pegs = [list(p) for p in pegs]
                    new_pegs[from_peg] = new_pegs[from_peg][1:]
                    new_pegs[to_peg] = [disk] + new_pegs[to_peg]
                    new_state = tuple(tuple(p) for p in new_pegs)
                    if new_state in state_set:
                        edge_key = tuple(sorted([state, new_state]))
                        if edge_key not in seen:
                            seen.add(edge_key)
                            edges.append((state, new_state, disk))
        return edges

    def _get_disk_y(self, peg_idx: int) -> float:
        base_y = self.TOWER_CENTER[1] - 1.0 + self.BASE_HEIGHT / 2 + self.DISK_HEIGHT / 2
        return base_y + len(self.peg_stacks[peg_idx]) * self.DISK_HEIGHT

    def _get_peg_x(self, peg_idx: int) -> float:
        return self.TOWER_CENTER[0] + (peg_idx - 1) * self.PEG_SPACING

    def _animated_solve(self) -> None:
        """Solve with synchronized tower and graph animation."""
        moves = []

        def hanoi(n: int, src: int, tgt: int, aux: int) -> None:
            if n == 0:
                return
            hanoi(n - 1, src, aux, tgt)
            moves.append((src, tgt))
            hanoi(n - 1, aux, tgt, src)

        hanoi(self.NUM_DISKS, 0, 2, 1)

        for from_peg, to_peg in moves:
            self._animate_move(from_peg, to_peg)

    def _animate_move(self, from_peg: int, to_peg: int) -> None:
        """Animate a single move on both tower and graph."""
        # Get disk to move
        disk = self.peg_stacks[from_peg].pop()

        # Compute new state for graph
        pegs = [list(p) for p in self.current_state]
        disk_num = pegs[from_peg][0]
        pegs[from_peg] = pegs[from_peg][1:]
        pegs[to_peg] = [disk_num] + pegs[to_peg]
        new_state = tuple(tuple(p) for p in pegs)

        # Get positions
        from_x = self._get_peg_x(from_peg)
        to_x = self._get_peg_x(to_peg)
        to_y = self._get_disk_y(to_peg)
        lift_y = self.TOWER_CENTER[1] + self.LIFT_HEIGHT

        # Get edge
        edge = self.edge_lines.get((self.current_state, new_state))

        # Animate: lift
        self.play(
            disk.animate.move_to([from_x, lift_y, 0]),
            run_time=self.ANIMATION_SPEED * 0.6,
        )

        # Animate: move horizontally + highlight edge + move marker
        anims = [disk.animate.move_to([to_x, lift_y, 0])]
        if edge:
            anims.append(edge.animate.set_stroke(opacity=1, width=3))
        anims.append(self.marker.animate.move_to(self.positions[new_state]))

        self.play(*anims, run_time=self.ANIMATION_SPEED * 0.6)

        # Animate: lower
        self.play(
            disk.animate.move_to([to_x, to_y, 0]),
            run_time=self.ANIMATION_SPEED * 0.6,
        )

        # Update state
        self.peg_stacks[to_peg].append(disk)
        self.current_state = new_state

        # Update counter
        self.move_count += 1
        self.remove(self.move_text)
        self.move_text = Text(f"Move: {self.move_count}", font_size=18, color=YELLOW)
        self.move_text.to_corner(DL, buff=0.4)
        self.add(self.move_text)


if __name__ == "__main__":
    pass
