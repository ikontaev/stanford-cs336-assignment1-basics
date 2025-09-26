class ListNode:
    """Node for doubly linked list representation of token sequence."""

    def __init__(self, token: int):
        self.token = token
        self.prev: ListNode | None = None
        self.next: ListNode | None = None

    def __repr__(self):
        return f"Node({self.token})"

    def __eq__(self, other):
        if not isinstance(other, ListNode):
            return False
        return self is other

    def __hash__(self):
        return hash(id(self))


class DoublyLinkedList:
    """Doubly linked list for efficient in-place token sequence modifications."""

    def __init__(self, tokens: list[int]):
        self.head: ListNode | None = None
        self.tail: ListNode | None = None
        self.size = 0

        # Build the linked list from token sequence
        for token in tokens:
            self.append(token)

    def append(self, token: int) -> ListNode:
        """Add a token to the end of the list and return the new node."""
        new_node = ListNode(token)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        return new_node

    def remove_node(self, node: ListNode) -> None:
        """Remove a node from the list in O(1) time."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        self.size -= 1

    def insert_after(self, node: ListNode, token: int) -> ListNode:
        """Insert a new token after the given node and return the new node."""
        new_node = ListNode(token)
        new_node.next = node.next
        new_node.prev = node

        if node.next:
            node.next.prev = new_node
        else:
            self.tail = new_node

        node.next = new_node
        self.size += 1
        return new_node

    def to_list(self) -> list[int]:
        """Convert linked list back to regular list for debugging/output."""
        result = []
        current = self.head
        while current:
            result.append(current.token)
            current = current.next
        return result
