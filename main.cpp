#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_CommHelpers.hpp>

int main (int argc, char *argv[]) {
    Tpetra::ScopeGuard tpetraScope (&argc, &argv);
    { // Scope guard, to make sure the destructor of Tpetra or Kokkos objects is called befor mpi and kokkos finalize
        auto comm = Tpetra::getDefaultComm ();

        using Teuchos::Array;
        using Teuchos::ArrayRCP;
        using Teuchos::ArrayView;
        using Teuchos::outArg;
        using Teuchos::RCP;
        using Teuchos::rcp;
        using Teuchos::REDUCE_SUM;
        using Teuchos::reduceAll;
        const int myRank = comm->getRank ();

        //*****************************************************
        // Define some shortcuts
        using map_type = Tpetra::Map<>;
        using vector_type = Tpetra::Vector<double>;
        using global_ordinal_type = vector_type::global_ordinal_type;
        using memory_space = vector_type::device_type::memory_space;

        //*****************************************************
        // Make sure each mpi rank has 5 entries
        const Tpetra::global_size_t numGlobalEntries = comm->getSize () * 5;

        //*****************************************************
        // Choose index style, here we choose to start with 0
        const global_ordinal_type indexBase = 0;

        //*****************************************************
        // Create a map
        RCP<const map_type> contigMap = rcp (new map_type (numGlobalEntries, indexBase, comm));

        //*****************************************************
        // Create a vector
        vector_type x (contigMap);

        //*****************************************************
        // Fill the vector with a value
        x.putScalar (42.0);

        //*****************************************************
        // Calc norm of the vector
        auto x_norm2 = x.norm2 ();
        std::cout << "Norm of x = " << x_norm2 << std::endl;

        //*****************************************************
        // Read values
        {
            x.sync_host ();
            auto x_2d = x.getLocalViewHost ();
            auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);

            const size_t localLength = x.getLocalLength ();
            for (size_t k = 0; k < localLength; ++k) {
                std::cout << x_1d(k) << std::endl;
            }
        }

        //*****************************************************
        // Modify values
        {
            x.sync_host ();
            auto x_2d = x.getLocalViewHost ();
            auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
            x.modify_host (); // Mark as modified

            const size_t localLength = x.getLocalLength ();
            for (size_t k = 0; k < localLength; ++k) {
                x_1d(k) = double (k); // Convert the variable k to double and write to the vector
            }

            x.sync<memory_space> (); // Sync to put modified values back
        }


        // Tell the Trilinos test framework that the test passed.
        if (comm->getRank () == 0) {
            std::cout << "End Result: TEST PASSED" << std::endl;
        }
    }
    return 0;
}
