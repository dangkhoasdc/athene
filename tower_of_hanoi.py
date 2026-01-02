"""
Tower of Hanoi Animation using Manim.

This module implements an animated visualization of the Tower of Hanoi puzzle
with N=4 disks using the Manim framework.
"""

from typing import List
from manim import *


class TowerOfHanoi(Scene):
    """Animated visualization of the Tower of Hanoi puzzle."""

    # Configuration
    NUM_DISKS = 4
    PEG_HEIGHT = 3.0
    PEG_WIDTH = 0.1
    BASE_HEIGHT = 0.15
    BASE_WIDTH = 2.0
    DISK_HEIGHT = 0.35
    MIN_DISK_WIDTH = 0.5
    MAX_DISK_WIDTH = 1.8
    PEG_SPACING = 4.0
    LIFT_HEIGHT = 2.2
    ANIMATION_TIME = 0.3

    # Colors for disks (from smallest to largest)
    DISK_COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE, PINK, TEAL]

    def construct(self) -> None:
        """Main construction method for the scene."""
        # Initialize data structures
        self.pegs: List[VGroup] = []  # Visual peg groups
        self.peg_stacks: List[List[VMobject]] = [[], [], []]  # Disks on each peg

        # Create and display title
        title = Text("Tower of Hanoi", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Create subtitle showing number of disks
        subtitle = Text(f"N = {self.NUM_DISKS} disks", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle))

        # Build the puzzle setup
        self._create_pegs()
        self._create_disks()

        self.wait(0.5)

        # Solve the puzzle with animation
        self._solve_hanoi(self.NUM_DISKS, 0, 2, 1)

        # Show completion message
        complete_text = Text("Solved!", font_size=48, color=GREEN)
        complete_text.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(complete_text, scale=1.5))
        self.wait(2)

    def _create_pegs(self) -> None:
        """Create the three pegs with bases."""
        peg_positions = [
            -self.PEG_SPACING,  # Left peg
            0,                   # Middle peg
            self.PEG_SPACING,    # Right peg
        ]

        peg_labels = ["Source", "Extra", "Destination"]

        for i, x_pos in enumerate(peg_positions):
            # Create peg (vertical rod)
            peg = Rectangle(
                height=self.PEG_HEIGHT,
                width=self.PEG_WIDTH,
                fill_color=GRAY_BROWN,
                fill_opacity=1,
                stroke_color=GRAY_BROWN,
            )
            peg.move_to([x_pos, self.PEG_HEIGHT / 2 - 1.5, 0])

            # Create base
            base = Rectangle(
                height=self.BASE_HEIGHT,
                width=self.BASE_WIDTH,
                fill_color=GRAY_BROWN,
                fill_opacity=1,
                stroke_color=GRAY_BROWN,
            )
            base.move_to([x_pos, -1.5, 0])

            # Create label
            label = Text(peg_labels[i], font_size=24, color=WHITE)
            label.next_to(base, DOWN, buff=0.2)

            # Group peg components
            peg_group = VGroup(peg, base, label)
            self.pegs.append(peg_group)
            self.play(FadeIn(peg_group), run_time=0.3)

    def _create_disks(self) -> None:
        """Create and place all disks on the first peg."""
        # Calculate disk widths (largest to smallest for stacking)
        disk_widths = []
        for i in range(self.NUM_DISKS):
            # Linear interpolation from max to min width
            width = self.MAX_DISK_WIDTH - (self.MAX_DISK_WIDTH - self.MIN_DISK_WIDTH) * i / max(1, self.NUM_DISKS - 1)
            disk_widths.append(width)

        # Create disks from largest (bottom) to smallest (top)
        for i in range(self.NUM_DISKS):
            disk = self._create_disk(disk_widths[i], self.DISK_COLORS[i % len(self.DISK_COLORS)])
            
            # Position disk on first peg
            y_pos = self._get_disk_y_position(0)
            x_pos = -self.PEG_SPACING
            disk.move_to([x_pos, y_pos, 0])
            
            self.peg_stacks[0].append(disk)
            self.play(FadeIn(disk, shift=DOWN), run_time=0.2)

    def _create_disk(self, width: float, color: ManimColor) -> RoundedRectangle:
        """Create a single disk with the specified width and color."""
        disk = RoundedRectangle(
            height=self.DISK_HEIGHT,
            width=width,
            corner_radius=0.1,
            fill_color=color,
            fill_opacity=1,
            stroke_color=WHITE,
            stroke_width=2,
        )
        return disk

    def _get_disk_y_position(self, peg_index: int) -> float:
        """Calculate the Y position for the next disk on a peg."""
        base_y = -1.5 + self.BASE_HEIGHT / 2 + self.DISK_HEIGHT / 2
        num_disks = len(self.peg_stacks[peg_index])
        return base_y + num_disks * self.DISK_HEIGHT

    def _get_peg_x_position(self, peg_index: int) -> float:
        """Get the X position of a peg."""
        return (peg_index - 1) * self.PEG_SPACING

    def _move_disk(self, from_peg: int, to_peg: int) -> None:
        """Animate moving a disk from one peg to another."""
        if not self.peg_stacks[from_peg]:
            return

        # Get the disk to move
        disk = self.peg_stacks[from_peg].pop()

        # Calculate positions
        from_x = self._get_peg_x_position(from_peg)
        to_x = self._get_peg_x_position(to_peg)
        to_y = self._get_disk_y_position(to_peg)

        # Animation sequence: lift, move horizontally, lower
        # 1. Lift the disk
        self.play(
            disk.animate.move_to([from_x, self.LIFT_HEIGHT, 0]),
            run_time=self.ANIMATION_TIME,
        )

        # 2. Move horizontally
        self.play(
            disk.animate.move_to([to_x, self.LIFT_HEIGHT, 0]),
            run_time=self.ANIMATION_TIME,
        )

        # 3. Lower to position
        self.play(
            disk.animate.move_to([to_x, to_y, 0]),
            run_time=self.ANIMATION_TIME,
        )

        # Update the data structure
        self.peg_stacks[to_peg].append(disk)

    def _solve_hanoi(self, n: int, source: int, target: int, auxiliary: int) -> None:
        """
        Recursively solve Tower of Hanoi and animate each move.

        Args:
            n: Number of disks to move
            source: Index of the source peg (0, 1, or 2)
            target: Index of the target peg (0, 1, or 2)
            auxiliary: Index of the auxiliary peg (0, 1, or 2)
        """
        if n == 0:
            return

        # Move n-1 disks from source to auxiliary
        self._solve_hanoi(n - 1, source, auxiliary, target)

        # Move the largest disk from source to target
        self._move_disk(source, target)

        # Move n-1 disks from auxiliary to target
        self._solve_hanoi(n - 1, auxiliary, target, source)


class TowerOfHanoiWithCounter(TowerOfHanoi):
    """Tower of Hanoi with a move counter display."""

    def construct(self) -> None:
        """Main construction method with move counter."""
        self.move_count = 0
        self.counter_text: Text = None

        # Initialize data structures
        self.pegs = []
        self.peg_stacks = [[], [], []]

        # Create and display title
        title = Text("Tower of Hanoi", font_size=36)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # Create subtitle
        subtitle = Text(f"N = {self.NUM_DISKS} disks", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle))

        # Create move counter
        self.counter_text = Text(f"Moves: 0", font_size=28, color=YELLOW)
        self.counter_text.to_edge(DOWN, buff=0.5)
        self.add(self.counter_text)

        # Build the puzzle setup
        self._create_pegs()
        self._create_disks()

        self.wait(0.5)

        # Solve the puzzle
        self._solve_hanoi(self.NUM_DISKS, 0, 2, 1)

        # Show completion
        total_moves = 2**self.NUM_DISKS - 1
        complete_text = Text(
            f"Solved in {total_moves} moves! (2^{self.NUM_DISKS} - 1)",
            font_size=32,
            color=GREEN,
        )
        complete_text.to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(self.counter_text, complete_text))
        self.wait(2)

    def _move_disk(self, from_peg: int, to_peg: int) -> None:
        """Override to update move counter."""
        super()._move_disk(from_peg, to_peg)

        # Update counter
        self.move_count += 1
        new_counter = Text(f"Moves: {self.move_count}", font_size=28, color=YELLOW)
        new_counter.to_edge(DOWN, buff=0.5)
        self.play(
            ReplacementTransform(self.counter_text, new_counter),
            run_time=0.1,
        )
        self.counter_text = new_counter


if __name__ == "__main__":
    # This allows running directly with: python tower_of_hanoi.py
    # But typically you'd use: manim -pql tower_of_hanoi.py TowerOfHanoi
    pass
